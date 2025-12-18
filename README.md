# Conformal Data Contamination Tests for Trading or Sharing of Data
This repository contains the code library used to create the results of the paper `Conformal Data Contamination Tests for Trading or Sharing of Data` submitted to NeurIPS'25.

## Contents
### Modules
- Benjamini_Hochberg.py `Base functionality to run adaptive Benjamini-Hochberg procedure.`
- ConformalContaminationTestModule.py `Class for computing conformal data contamination test statistics and p-values.`
- ConformalScoreModule.py `Class for computing conformal scores.`
- DataHandlerModule.py `Class for loading and organizing the MNIST and FEMNIST data.`
- SupervisedMachineLearningModule.py `Class for fitting and evaluating classifiers.`
- utilities.py `Various functionality used in main scripts.`

### Simulation scripts
- BudgetAccuracy.py `Run a simulation study with the proposed procedure and the baselines using a fixed budget and computing classification accuracies - See Section 4.`
- BudgetAccuracyCV.py `Run a simulation study with the proposed procedure and the baselines selecting the budget based on the data in the first round - See Section S4.`
- ScoringAnalysis.py `Run a simulation study with the proposed procedure and the baselines evaluating only the conformal data contamination tests.`

### Recreating figures and table
- Figures 2 and 8: ScoringResults/ScoringResultsLoad.py
- Figure 3: BudgetAccuracy/BudgetAccuracyLoad.py
- Table 1: ScoringResults/ScoringResultsLoad.py
- Table 4: BudgetAccuracyCV/BudgetAccuracyCVLoad.py

## Software Setup

### Python dependencies
```
python 3
numpy
matplotlib
pandas
scipy
scikit-learn
pytorch
```

## Data Setup

### FEMNIST
- Step 1: Download `femnist.tar.gz` from `https://github.com/GwenLegate/femnist-dataset-PyTorch`.
- Step 2: Unzip `femnist.tar.gz` yielding a folder `femnist` with files `femnist_test.pt`, `femnist_train.pt`, and `femnist_user_keys.pt`.
- Step 3: Move `femnist_test.pt` and `femnist_train.pt` to the `FEMNIST` folder in this repo.
- Step 4: Execute the `femnist_pt_to_npy.py` script in the `FEMNIST` folder to have the data in the desired format.

### MNIST
- Included in this repo in folder `MNIST_CSV`.
