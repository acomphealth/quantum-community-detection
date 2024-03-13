from functions.helper import load_karate, load_syndata, evaluate_partition_hybrid
import os
import shutil
import wandb


with wandb.init( 
        project="quantum-clustering",
        # Track hyperparameters and run metadata
        config={
            "num_parts": 5,
            "dataset": "data/stdmerge-n32-q8-pout01/stdmerge-n32-q8-pout01.t00100.graph",
            "run_label": "syndata",
            "qsize": 64,
            "threshold": 0,
            "beta0": 1,
            "gamma0": -5
        }) as run:
    os.makedirs("results", exist_ok=True)
    graph = load_syndata("data/stdmerge-n32-q8-pout01/stdmerge-n32-q8-pout01.t00100.graph")
    evaluate_partition_hybrid(
            num_parts=run.config.num_parts,
            graph=graph,
            dataset=run.config.dataset,
            run_label=run.config.run_label,
            qsize=run.config.qsize,
            threshold=run.config.threshold,
            beta0=run.config.beta0,
            gamma0=run.config.gamma0,
            run_profile="defaults"
        )

wandb.finish()
shutil.rmtree("results")