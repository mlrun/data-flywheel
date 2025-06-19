import mlrun
from datetime import datetime
from src.api.models import DataSplitConfig, FlywheelRun, TaskResult
from src.tasks.tasks import create_datasets
from src.lib.nemo.llm_as_judge import LLMAsJudge

def create_dataset(
    context: mlrun.MLClientCtx,
    workload_id: str,
    client_id: str,
    data_split_config: dict = None,
):
    """
    Create datasets for a given workload and client ID.

    :param context:           MLRun context.
    :param workload_id:       ID of the workload.
    :param client_id:         ID of the client.
    :param data_split_config: Configuration for data splitting.
                              If None, will use the default configuration.

    :return: A JSON representation of the TaskResult containing the created datasets and other metadata.
    """
    llm_as_judge = LLMAsJudge()
    llm_as_judge_cfg = llm_as_judge.config
    split_config = DataSplitConfig(**data_split_config) if data_split_config else None
    flywheel_run = FlywheelRun(
        workload_id=workload_id,
        client_id=client_id,
        started_at=datetime.utcnow(),
        num_records=0,  # Will be updated when datasets are created
        nims=[],
    )
    previous_result = TaskResult(
        workload_id=workload_id,
        flywheel_run_id=str(flywheel_run.id),
        client_id=client_id,
        error=None,  # Reset any previous errors
        datasets={},
        llm_judge_config=llm_as_judge_cfg,
        data_split_config=split_config,
    )
    return create_datasets(previous_result=previous_result)
