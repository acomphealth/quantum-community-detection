import ray
import argparse
from ray import train, tune
from driver import run_experiment
import os
from azure.ai.ml import MLClient
import mlflow
from azure.identity import DefaultAzureCredential


parser = argparse.ArgumentParser()
parser.add_argument("--address")
parser.add_argument("--parent-run-id", required=False, default=None)
args = parser.parse_args()
ray.init(address=args.address)

parent_run_id = None

base_path = "/home/wls9/project/quantum-community-detection/data"
storage_path = os.path.expanduser("~/ray_results")
azml_experiment_name = "quantum-community-detection"
ray_experiment_name = "quantum-community-detection"
path = os.path.join(storage_path, ray_experiment_name)
azml_config_path = "/home/wls9/project/quantum-community-detection/"

if args.parent_run_id is None:
    os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "3000"
    ml_client = MLClient.from_config(path=azml_config_path, credential=DefaultAzureCredential())
    workspace = ml_client.workspaces.get(ml_client.workspace_name)
    azureml_mlflow_uri = workspace.mlflow_tracking_uri
    mlflow.set_tracking_uri(azureml_mlflow_uri)

    mlflow.set_experiment(azml_experiment_name)
    experiment_id = mlflow.get_experiment_by_name(azml_experiment_name).experiment_id
    dashboard = f"https://ml.azure.com/experiments/id/{experiment_id}"

    mlflow.start_run(run_name=ray_experiment_name)
    parent_run_id = mlflow.active_run().info.run_id
else:
    parent_run_id = args.parent_run_id

search_space = {
    "num_parts": tune.grid_search([8, 7, 6, 5, 4, 3]),
    "qsize": tune.grid_search([32, 64, 128]),
    "threshold": tune.grid_search([0, 0.05, 0.1, 0.2, 0.3, 0.5]),
    "beta0": tune.grid_search([-50, -10, -5, -1, 0, 1, 5, 10, 50]),
    "gamma0": tune.grid_search([-250, -50, -25, -5, 0, 5, 25, 50, 250]),
    "base_path": tune.grid_search([base_path]),
    "graph_name": tune.grid_search(["stdmerge-n32-q8-pout01"]),
    "graph_iter": tune.grid_search(["t00100"]),
    "azure_config_path": tune.grid_search([azml_config_path]),
    "azml_experiment_name": tune.grid_search([azml_experiment_name]),
    "ray_experiment_name": tune.grid_search([ray_experiment_name]),
    "parent_run_id": tune.grid_search([parent_run_id])
}

trainable_with_resources = tune.with_resources(run_experiment, {"cpu": 5})

tuner = None
if tune.Tuner.can_restore(path):
    print("!!! Restoring prior experiment from {path}")
    tuner = tune.Tuner.restore(path, trainable=trainable_with_resources, param_space=search_space)
else:
    tuner = tune.Tuner(  # ③
        trainable_with_resources,
        param_space=search_space,
        run_config=train.RunConfig(
            name=ray_experiment_name,
            storage_path="/home/wls9/project/ray-path",
        ),
        tune_config=tune.TuneConfig(max_concurrent_trials=1)
    )

results = tuner.fit()

mlflow.end_run()
