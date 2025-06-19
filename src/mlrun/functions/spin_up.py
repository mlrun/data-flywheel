from datetime import datetime

from bson import ObjectId

from src.api.db_manager import TaskDBManager
from src.api.models import (
    NIMRun,
    NIMRunStatus,
    TaskResult,

)
from src.api.schemas import DeploymentStatus
from src.config import NIMConfig, settings

from src.lib.nemo.dms_client import DMSClient
from src.log_utils import setup_logging
import mlrun

logger = setup_logging("data_flywheel.tasks")

# Centralised DB helper - keeps Mongo specifics out of individual tasks
db_manager = TaskDBManager()


def spin_up_nim(context: mlrun.MLClientCtx, previous_result: dict, nim_config: dict) -> TaskResult:
    previous_result = TaskResult(**previous_result)

    ## reset previous_result.error as new nim starts
    previous_result.error = None
    if isinstance(nim_config, str):
        # If nim_config is a string, assume it's a JSON string and parse it
        import json
        nim_config = json.loads(nim_config)
    nim_config = NIMConfig(**nim_config)
    previous_result.nim = nim_config

    # Create NIM run in the nims collection
    start_time = datetime.utcnow()
    nim_run = NIMRun(
        flywheel_run_id=ObjectId(previous_result.flywheel_run_id),
        model_name=nim_config.model_name,
        evaluations=[],
        started_at=start_time,
        finished_at=start_time,  # Will be updated when evaluations complete
        runtime_seconds=0,  # Will be updated when evaluations complete
    )

    # Persist and mark status via DB manager
    nim_run.id = db_manager.create_nim_run(nim_run)
    db_manager.set_nim_status(nim_run.id, NIMRunStatus.DEPLOYING)

    try:
        dms_client = DMSClient(nmp_config=settings.nmp_config, nim=nim_config)

        if not dms_client.is_deployed():
            logger.info(f"Deploying NIM {nim_config.model_name}")

            try:
                dms_client.deploy_model()
            except Exception as e:
                logger.error(f"Error deploying NIM {nim_config.model_name}: {e}")
                db_manager.set_nim_status(nim_run.id, NIMRunStatus.ERROR, error=str(e))
                previous_result.error = str(e)
                return previous_result.model_dump()
        else:
            logger.info(f"NIM {nim_config.model_name} is already deployed")

        def progress_callback(status: dict):
            current_time = datetime.utcnow()
            db_manager.update_nim_deployment_status(
                nim_run.id,
                DeploymentStatus(status.get("status", "unknown")),
                runtime_seconds=(current_time - start_time).total_seconds()
            )

        dms_client.wait_for_deployment(progress_callback=progress_callback)

        dms_client.wait_for_model_sync(nim_config.target_model_for_evaluation())

        db_manager.set_nim_status(nim_run.id, NIMRunStatus.RUNNING)

        return previous_result.model_dump()
    except Exception as e:
        error_msg = f"Error spinning up NIM: {e!s}"
        logger.error(error_msg)
        # Persist error on NIM run
        db_manager.set_nim_status(
            nim_run.id,
            NIMRunStatus.ERROR,
            error=error_msg,
            deployment_status=DeploymentStatus.FAILED,
        )
        dms_client.shutdown_deployment()
        previous_result.error = error_msg

        return previous_result.model_dump()
