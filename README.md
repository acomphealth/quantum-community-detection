# Setup and Prerequisites
## Conda
Conda will need to be installed, then create the virtual environment with the provided manifest:

```
conda env create -f environment.yml
```

After your environment is created, activate it with:

```
conda activate cm4ai-quantum
```

## Weights and Biases
Create an account with [Weights & Biases](https://wandb.ai/) if you do not already have one. Then get your API key from the Settings menu. After you have your key, run the following:

```
wandb login
```

## D-Wave
Register for an account with [D-Wave](https://cloud.dwavesys.com/leap/) (free trial). After you have an account, obtain your Solver API Token from the menu. Then run:

```
dwave config create
```

# Running the Current Example
After running the above setup and configuration steps, you can run a predefined set of parameters against a synthetic, benchmark graph from the root of this project with:

```
python src/driver.py
```