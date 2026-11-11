# Machine Learning Assignment 3: CIFAR-10

**Aman Kumar | M26MAC001**

This repository contains the completed Q1-Q4 notebooks, saved split and experiment settings, training histories, final result tables, figures, and model checkpoints. The assignment report is submitted separately.

## Contents

- `Q1.ipynb`-`Q4.ipynb`: executable notebooks, saved cell outputs, explanations, and conclusions.
- `configurations.json`: a human-readable summary of the settings. The notebooks are the source of truth for the implementation.
- `results/`, `results_q2/`, `results_q3/`, `results_q4/`: saved splits, histories, result tables, figures, trained weights, and selected models.
- `final_logs/`: text extracted from the saved notebook outputs, including progress and results.
- `environment_windows_snapshot.txt`: package names and versions observed in the original `.venv` on 11 Nov 2026. This snapshot was reconstructed from installed package metadata, not from a saved historical `pip freeze`.
- `scripts/run_notebooks.py`: executes all code cells in order without modifying notebooks, or checks that they compile.
- `scripts/plot_results.py`: regenerates learning curves and SVM plots from saved CSV files; when CIFAR-10 is present, it also rebuilds Q4 boundary-image grids. The notebooks contain the original code for every reported figure, including the Q3 reconstruction grid.

The raw CIFAR-10 download and `.venv` are deliberately not included. CIFAR-10 is automatically downloaded by torchvision when the notebooks run. The split is fixed with seed **26001** from the roll-number digits; 45,000 training and 5,000 validation images are selected by stratification. `results/split_seed_26001.npz` saves the split and `results_q4/fixed_indices.npz` saves the fixed Q4 subset.

## Environment and setup (Windows PowerShell)

The recorded environment was Windows with Python **3.15.2** and the packages in `environment_windows_snapshot.txt`. An internet connection is needed for packages and the first CIFAR-10 download. From this folder:

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r environment_windows_snapshot.txt
```

Open the repository folder in VS Code, select `.venv` as the Jupyter kernel, and run `Q1.ipynb`, then `Q2.ipynb`, `Q3.ipynb`, and `Q4.ipynb`. Q3 loads the Q2 clean CNN checkpoint; Q4 loads the validation-selected Q3 encoder. The original results and checkpoints are included.

## Reproduction commands

Run these commands from this directory. **Make a copy of the repository before a full rerun**: training cells replace files in the result folders. Full CPU training can take many hours. A CUDA-enabled PyTorch installation and supported GPU can reduce training time, but can yield slightly different numerical results.

```powershell
# Check every code cell without running training:
.\.venv\Scripts\python.exe scripts\run_notebooks.py --check

# Re-run all four notebooks in order and save a fresh console log:
New-Item -ItemType Directory -Force logs | Out-Null
.\.venv\Scripts\python.exe -u scripts\run_notebooks.py 2>&1 | Tee-Object -FilePath logs\reproduction.txt

# Recreate the CSV-based figures; boundary-image grids need downloaded CIFAR-10:
.\.venv\Scripts\python.exe scripts\plot_results.py
```

To reproduce a single question after its dependencies exist, use for example `.\.venv\Scripts\python.exe -u scripts\run_notebooks.py --question 3`. Executing notebook cells from top to bottom in VS Code is the other supported route.

The official test set must not be used to tune models. The code selects checkpoints and hyperparameters using validation data before final test evaluation. The Q4 class-pair analysis uses test predictions descriptively after the SVM is fixed.

## Reproducibility limits

The saved CSVs and notebook outputs are the reported original run. Random seeds and deterministic cuDNN options are set, but exact bitwise agreement is not guaranteed across versions, operating systems, or hardware. The project environment snapshot describes packages installed when the repository was prepared; it cannot independently prove the exact versions at the time each original run occurred. Preserve the saved outputs when comparing reruns.

CIFAR-10: Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009; https://www.cs.toronto.edu/~kriz/cifar.html.
