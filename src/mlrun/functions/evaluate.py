from datetime import datetime
from typing import Any

from src.api.db_manager import TaskDBManager
from src.api.models import (
    DatasetType,
    EvalType,
    EvaluationResult,
    NIMEvaluation,
    TaskResult,
    ToolEvalType,
    WorkloadClassification,
)
from src.config import settings
from src.lib.nemo.evaluator import Evaluator
from src.log_utils import setup_logging
import mlrun

logger = setup_logging("data_flywheel.tasks")

# Centralised DB helper - keeps Mongo specifics out of individual tasks
db_manager = TaskDBManager()

def run_generic_eval(
    previous_result: TaskResult, eval_type: EvalType, dataset_type: DatasetType
) -> dict:
    """
    Run the Base/ICL/Customization evaluation against the NIM based on the eval_type.
    Takes the NIM details from the previous task.
    """
    if _should_skip_stage(previous_result, f"run_{eval_type}_eval"):
        return previous_result.model_dump()

    logger.info(f"Running {eval_type} evaluation")
    llm_judge_config = previous_result.llm_judge_config
    evaluator = Evaluator(judge_model_config=llm_judge_config.judge_model_config())
    start_time = datetime.utcnow()

    tool_eval_types = [None]
    if previous_result.workload_type == WorkloadClassification.TOOL_CALLING:
        tool_eval_types = [ToolEvalType.TOOL_CALLING_METRIC]  # , ToolEvalType.TOOL_CALLING_JUDGE]

    jobs: list[dict[str, Any]] = []

    for tool_eval_type in tool_eval_types:
        # Find the NIM run for this model
        nim_run = db_manager.find_nim_run(
            previous_result.flywheel_run_id,
            previous_result.nim.model_name,
        )
        if not nim_run:
            msg = f"No NIM run found for model {previous_result.nim.model_name}"
            logger.error(msg)
            raise ValueError(msg)

        # Create evaluation document first
        evaluation = NIMEvaluation(
            nim_id=nim_run["_id"],
            eval_type=eval_type,
            scores={},  # Will be updated when evaluation completes
            started_at=start_time,
            finished_at=None,  # Will be updated when evaluation completes
            runtime_seconds=0.0,  # Will be updated when evaluation completes
            progress=0.0,  # Will be updated during evaluation
        )

        # Add evaluation to the database
        db_manager.insert_evaluation(evaluation)

        # Fix: Create closure with bound variables
        def make_progress_callback(manager: TaskDBManager, eval_instance):
            def callback(update_data):
                """Update evaluation document with progress"""
                manager.update_evaluation(eval_instance.id, update_data)

            return callback

        # Create callback with properly bound variables
        progress_callback = make_progress_callback(db_manager, evaluation)

        # Run the evaluation
        try:
            # Use customized model name for customization evaluation
            target_model = (
                previous_result.customization.model_name
                if eval_type == EvalType.CUSTOMIZED
                else previous_result.nim.target_model_for_evaluation()
            )

            job_id = evaluator.run_evaluation(
                dataset_name=previous_result.datasets[dataset_type],
                workload_type=previous_result.workload_type,
                target_model=target_model,  # Use the selected target model
                test_file="eval_data.jsonl",
                tool_eval_type=tool_eval_type,
                limit=settings.data_split_config.limit,
            )
            logger.info("Evaluation job id: %s", job_id)

            # update uri in evaluation
            evaluation.nmp_uri = evaluator.get_job_uri(job_id)
            progress_callback({"nmp_uri": evaluation.nmp_uri})

            jobs.append(
                {
                    "job_id": job_id,
                    "evaluation": evaluation,
                    "progress_callback": progress_callback,
                    "tool_eval_type": tool_eval_type,
                }
            )
        except Exception as e:
            error_msg = f"Error running {eval_type} evaluation: {e!s}"
            logger.error(error_msg)
            db_manager.update_evaluation(
                evaluation.id,
                {
                    "error": error_msg,
                    "finished_at": datetime.utcnow(),
                    "progress": 0.0,
                },
            )
            previous_result.error = error_msg
            return previous_result.model_dump()

    for job in jobs:
        # Wait for completion with progress updates
        try:
            evaluator.wait_for_evaluation(
                job_id=job["job_id"],
                flywheel_run_id=previous_result.flywheel_run_id,
                polling_interval=5,
                timeout=3600,
                progress_callback=job["progress_callback"],
            )

            # Get final results
            results = evaluator.get_evaluation_results(job["job_id"])
            logger.info(results)

            # Update final results
            finished_time = datetime.utcnow()
            scores: dict[str, float] = {}
            if previous_result.workload_type == WorkloadClassification.TOOL_CALLING:
                if results["tasks"]["custom-tool-calling"]:
                    scores["function_name"] = results["tasks"]["custom-tool-calling"]["metrics"][
                        "tool-calling-accuracy"
                    ]["scores"]["function_name_accuracy"]["value"]
                    scores["function_name_and_args_accuracy"] = results["tasks"][
                        "custom-tool-calling"
                    ]["metrics"]["tool-calling-accuracy"]["scores"][
                        "function_name_and_args_accuracy"
                    ]["value"]

                if results["tasks"]["custom-tool-calling"]["metrics"]["correctness"]:
                    scores["tool_calling_correctness"] = results["tasks"]["custom-tool-calling"][
                        "metrics"
                    ]["correctness"]["scores"]["rating"]["value"]
            else:
                scores["similarity"] = results["tasks"]["llm-as-judge"]["metrics"]["llm-judge"][
                    "scores"
                ]["similarity"]["value"]

            job["progress_callback"](
                {
                    "scores": scores,
                    "finished_at": finished_time,
                    "runtime_seconds": (finished_time - start_time).total_seconds(),
                    "progress": 100.0,
                }
            )
            previous_result.add_evaluation(
                eval_type,
                EvaluationResult(
                    job_id=job["job_id"],
                    scores=scores,
                    started_at=start_time,
                    finished_at=finished_time,
                    percent_done=100.0,
                ),
            )
        except Exception as e:
            error_msg = f"Error running {eval_type} evaluation: {e!s}"
            logger.error(error_msg)
            db_manager.update_evaluation(
                job["evaluation"].id,
                {
                    "error": error_msg,
                    "finished_at": datetime.utcnow(),
                    "progress": 0.0,
                },
            )
            # no need to raise error here, the error is captured, let the task continue to spin down the deployment
            previous_result.error = error_msg

    return previous_result.model_dump(mode="json")

def run_base_eval(context: mlrun.MLClientCtx, previous_result: dict = None) -> dict:
    if previous_result is not None:
        previous_result = TaskResult(**previous_result)
    return run_generic_eval(previous_result, EvalType.BASE, DatasetType.BASE)

def run_icl_eval(context: mlrun.MLClientCtx, previous_result: dict = None) -> dict:
    if previous_result is not None:
        previous_result = TaskResult(**previous_result)
    return run_generic_eval(previous_result, EvalType.ICL, DatasetType.ICL)

def run_customization_eval(context: mlrun.MLClientCtx, previous_result: dict) -> dict:
    if previous_result is not None:
        previous_result = TaskResult(**previous_result)
    if _should_skip_stage(previous_result, "run_customization_eval"):
        return previous_result.model_dump()

    try:
        if not previous_result.nim.customization_enabled:
            logger.info(f"Customization disabled for {previous_result.nim.model_name}")
            return previous_result.model_dump()

        if not previous_result.customization or not previous_result.customization.model_name:
            msg = "No customized model available for evaluation"
            logger.error(msg)
            raise ValueError(msg)

        customization_model = previous_result.customization.model_name
        workload_id = previous_result.workload_id

        # Find the customization document
        customization_doc = db_manager.find_customization(
            workload_id,
            customization_model,
        )
        if not customization_doc:
            msg = f"No customization found for model {customization_model}"
            logger.error(msg)
            raise ValueError(msg)

        print(
            f"Running evaluation on customized model {customization_model} for workload {workload_id}"
        )

        next_result = run_generic_eval(previous_result, EvalType.CUSTOMIZED, DatasetType.BASE)

        return next_result
    except Exception as e:
        error_msg = f"Error running customization evaluation: {e!s}"
        logger.error(error_msg)
        previous_result.error = error_msg
        return previous_result.model_dump()

def _should_skip_stage(previous_result: TaskResult | None, stage_name: str) -> bool:
    if isinstance(previous_result, TaskResult) and previous_result.error:
        logger.warning(
            "Skipping %s because a previous stage failed: %s",
            stage_name,
            previous_result.error,
        )
        return True
    return False
