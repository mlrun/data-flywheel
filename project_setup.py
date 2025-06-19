# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import mlrun

LOCAL_REGISTRY = "192.168.49.2:5000"

def setup(
        project: mlrun.projects.MlrunProject,
) -> mlrun.projects.MlrunProject:
    ngc_api_key = os.getenv("NGC_API_KEY")
    source = project.get_param(key="source")
    registry = project.get_param(key="registry", default=LOCAL_REGISTRY)

    # This is a workaround for not running the setup if the project is already set up.
    # It happens when getting the project in the workflow.
    if not source:
        return project

    project.set_source(source, pull_at_runtime=True)
    project.set_default_image(f"{registry}/mlrun-data-flywheel:latest")
    if ngc_api_key:
        project.set_secrets(secrets={'NGC_API_KEY': ngc_api_key})

    _set_function(
        project=project,
        func="src/mlrun/functions/create_dataset.py",
        name="create-dataset",
        handler="create_dataset",
    )

    _set_function(
        project=project,
        func="src/mlrun/functions/wait_for_llm_as_a_judge.py",
        name="wait-for-llm-as-a-judge",
        handler="wait_for_llm_as_judge",
    )

    _set_function(
        project=project,
        func="src/mlrun/functions/spin_up.py",
        name="spin-up-nims",
        handler="spin_up_nim",
    )
    _set_function(
        project=project,
        func="src/mlrun/functions/evaluate.py",
        name="evaluate",
        handler="run_base_eval",
    )
    _set_function(
        project=project,
        func="src/mlrun/functions/customization.py",
        name="customize",
        handler="start_customization",
    )
    _set_function(
        project=project,
        func="src/mlrun/functions/shutdown_deployment.py",
        name="shutdown-deployment",
        handler="shutdown_deployment",
    )
    project.set_workflow(
        name="data-flywheel-job",
        workflow_path="src/mlrun/workflows/data_flywheel_workflow.py",
        image="mlrun/mlrun-kfp",
    )

    return project


def _set_function(
        project: mlrun.projects.MlrunProject,
        func: str,
        name: str,
        handler: str = None,
):
    fn = project.set_function(
        func=func,
        name=name,
        handler=handler,
        with_repo=True,
    )
    fn.set_envs(
        env_vars={
            "ELASTICSEARCH_URL": "http://host.minikube.internal:9200",
            "REDIS_URL": "redis://host.minikube.internal:6379/0",
            "MONGODB_URL": "mongodb://host.minikube.internal:27017",
            "MONGODB_DB": "flywheel",
        }
    )
    fn.save()
