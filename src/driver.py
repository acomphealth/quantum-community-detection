from functions.helper import load_syndata, evaluate_partition_hybrid
import os
import shutil
import warnings
import logging
from azure.ai.ml import MLClient
import mlflow
from azure.identity import DefaultAzureCredential


logger = logging.getLogger('azure')
logger.setLevel(logging.INFO)
# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_experiment(config):
    os.environ["AZUREML_ARTIFACTS_DEFAULT_TIMEOUT"] = "3000"
    ml_client = MLClient.from_config(path=config["azure_config_path"], credential=DefaultAzureCredential())
    workspace = ml_client.workspaces.get(ml_client.workspace_name)
    azureml_mlflow_uri = workspace.mlflow_tracking_uri
    mlflow.set_tracking_uri(azureml_mlflow_uri)

    azml_experiment_name = config["azml_experiment_name"]
    mlflow.set_experiment(azml_experiment_name)
    experiment_id = mlflow.get_experiment_by_name(azml_experiment_name).experiment_id
    dashboard = f"https://ml.azure.com/experiments/id/{experiment_id}"  # noqa: F841

    num_parts = config["num_parts"]
    qsize = config["qsize"]
    threshold = config["threshold"]
    beta0 = config["beta0"]
    gamma0 = config["gamma0"]
    base_path = config["base_path"]
    graph_name = config["graph_name"]
    graph_iteration = config["graph_iter"]

    datafile = os.path.join(base_path, f"{graph_name}/{graph_name}.{graph_iteration}.graph")
    ground_truth_file = os.path.join(base_path, f"{graph_name}/{graph_name}.{graph_iteration}.comms")

    with mlflow.start_run(run_id=config["parent_run_id"]) as parent_run:  # noqa: F841
        with mlflow.start_run(nested=True) as child_run:
            try:
                mlflow.log_params(config)
                run_id = child_run.info.run_id
                print(f"Running job ID {run_id}")

                os.makedirs(f"results/{run_id}", exist_ok=True)
                graph = load_syndata(f"data/{datafile}")
                evaluate_partition_hybrid(
                    num_parts=num_parts,
                    graph=graph,
                    ground_truth_path=ground_truth_file,
                    dataset=datafile,
                    run_label=run_id,
                    qsize=qsize,
                    threshold=threshold,
                    beta0=beta0,
                    gamma0=gamma0,
                    run_profile="defaults",
                    run_id=run_id
                )

                shutil.rmtree(f'results/{run_id}')
                mlflow.end_run()

            except Exception as ex:
                mlflow.set_tag("LOG_STATUS", "FAILED")
                mlflow.log_text(str(ex), "exception.txt")
                mlflow.end_run(status="FAILED")
                raise ex
