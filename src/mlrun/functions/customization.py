from datetime import datetime

from src.api.db_manager import TaskDBManager
from src.api.models import (
    DatasetType,
    NIMCustomization,
    TaskResult,
)

from src.config import settings

from src.lib.nemo.customizer import Customizer
from src.log_utils import setup_logging
import mlrun

logger = setup_logging("data_flywheel.tasks")

# Centralised DB helper - keeps Mongo specifics out of individual tasks
db_manager = TaskDBManager()

def start_customization(context: mlrun.MLClientCtx, previous_result: dict) -> TaskResult:
    """
    Start customization process for the NIM.
    Takes the previous evaluation results.

    Args:
        previous_result: Result from the previous task containing workload_id and target_llm_model
    """
    previous_result = TaskResult(**previous_result)
    if _should_skip_stage(previous_result, "start_customization"):
        return previous_result.model_dump(mode="json")

    if not previous_result.nim.customization_enabled:
        logger.info(
            f"Customization skipped for {previous_result.nim.model_name} because it is using an external NIM"
        )
        return previous_result.model_dump(mode="json")

    workload_id = previous_result.workload_id
    target_llm_model = previous_result.nim.model_name
    logger.info(
        f"Starting NIM customization for workload {workload_id} on model {target_llm_model}"
    )

    # Find the NIM run
    nim_run = db_manager.find_nim_run(
        previous_result.flywheel_run_id,
        previous_result.nim.model_name,
    )
    if not nim_run:
        msg = f"No NIM run found for model {target_llm_model}"
        logger.error(msg)
        raise ValueError(msg)

    start_time = datetime.utcnow()
    customizer = Customizer()

    # Create customization document with training tracking fields
    customization = NIMCustomization(
        nim_id=nim_run["_id"],
        workload_id=workload_id,
        base_model=target_llm_model,
        customized_model=None,  # Will be set when job starts
        started_at=start_time,
        progress=0.0,
        epochs_completed=0,
        steps_completed=0,
    )

    # Add customization to database
    db_manager.insert_customization(customization)

    def progress_callback(update_data):
        """Update customization document with progress"""
        db_manager.update_customization(customization.id, update_data)

    output_model_name = f"customized-{target_llm_model}".replace("/", "-")

    try:
        # Start customization job
        customization_job_id, customized_model = customizer.start_training_job(
            name=f"customization-{workload_id}-{target_llm_model}",
            base_model=previous_result.nim.model_name,
            output_model_name=output_model_name,
            dataset_name=previous_result.datasets[DatasetType.TRAIN],
            training_config=settings.training_config,
        )
        logger.info(f"Customization job id: {customization_job_id}")

        # update uri in customization
        customization.nmp_uri = customizer.get_job_uri(customization_job_id)
        db_manager.update_customization(customization.id, {"nmp_uri": customization.nmp_uri})

        # Update customization with model name
        progress_callback({"customized_model": customized_model})

        # Wait for completion with progress updates
        customizer.wait_for_customization(
            customization_job_id,
            flywheel_run_id=previous_result.flywheel_run_id,
            progress_callback=progress_callback,
        )

        customizer.wait_for_model_sync(
            flywheel_run_id=previous_result.flywheel_run_id,
            customized_model=customized_model,
        )

        # Update completion status
        finished_time = datetime.utcnow()
        final_update = {
            "finished_at": finished_time,
            "runtime_seconds": (finished_time - start_time).total_seconds(),
            "progress": 100.0,
        }
        progress_callback(final_update)

        # Final TaskResult update
        previous_result.update_customization(
            job_id=customization_job_id,
            model_name=customized_model,
            started_at=start_time,
            finished_at=finished_time,
            percent_done=100.0,
        )

    except Exception as e:
        error_msg = f"Error starting customization: {e!s}"
        logger.error(error_msg)
        db_manager.update_customization(
            customization.id,
            {
                "error": error_msg,
                "finished_at": datetime.utcnow(),
                "progress": 0.0,
            },
        )
        previous_result.error = error_msg
    return previous_result.model_dump(mode="json")

def _should_skip_stage(previous_result: TaskResult | None, stage_name: str) -> bool:
    if isinstance(previous_result, TaskResult) and previous_result.error:
        logger.warning(
            "Skipping %s because a previous stage failed: %s",
            stage_name,
            previous_result.error,
        )
        return True
    return False