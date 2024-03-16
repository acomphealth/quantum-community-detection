from functions.helper import load_karate, load_syndata, evaluate_partition_hybrid
import os
import shutil
import wandb

datafile = "stdmerge-n32-q8-pout01/stdmerge-n32-q8-pout01.t00100.graph"
ground_truth_file = "data/stdmerge-n32-q8-pout01/stdmerge-n32-q8-pout01.t00100.comms"

with wandb.init() as run:
    os.makedirs("results", exist_ok=True)
    graph = load_syndata(f"data/{datafile}")
    evaluate_partition_hybrid(
            num_parts=run.config.num_parts,
            graph=graph,
            ground_truth_path=ground_truth_file,
            dataset=datafile,
            run_label=datafile,
            qsize=run.config.qsize,
            threshold=run.config.threshold,
            beta0=run.config.beta0,
            gamma0=run.config.gamma0,
            run_profile="defaults"
        )

wandb.finish()
shutil.rmtree("results")
