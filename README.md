# Conformal Data Contamination Tests for In-distribution Data Acquisition
This repository contains the code library used to create the results of the paper `Conformal Data Contamination Tests for In-distribution Data Acquisition`, Transactions on Machine Learning Research, 2026.

## Contents
### Modules
- Benjamini_Hochberg.py `Base functionality to run adaptive Benjamini-Hochberg procedure.`
- ConformalContaminationTestModule.py `Class for computing conformal data contamination test statistics and p-values.`
- ConformalScoreModule.py `Class for computing conformal scores.`
- DataHandlerModule.py `Class for loading, organizing, and sampling the data.`
- SupervisedMachineLearningModule.py `Class for fitting and evaluating classifiers.`
- utilities.py `Various functionality used in main scripts.`
- autoencoder.py `Base implementation of an autoencoder.`
- ResNet18.py `Base implementation of convolutional neural networks and residual networks.`

### Simulation scripts
- ProposedAccuracy.py `Run a simulation study with the proposed procedure and the baselines - See Section 4 and Section D4.`
- ProposedAccuracyCV.py `Run a simulation study with the proposed procedure and the baselines selecting hyperparameters based on the data in the first round - See Section D5.`
- ScoringAnalysis.py `Run a simulation study with the proposed procedure and the baselines evaluating only the conformal data contamination tests - See Section 4 and Section D4.`

### Recreating figures and table
- Figure 2: ScoringAnalysis/ScoringResultsLoad.py
- Table 1: ScoringAnalysis/ScoringBH.py
- Figure A5: ScoringAnalysis/CODplot.py
- Tables A3-A5: ScoringAnalysis/AUCtables.py
- Table A6: ScoringAnalysis/TDRtables.py
- Tables A7-A8: ScoringAnalysis/FDRtables.py
- Figures 3 & A6: ProposedAccuracy/ProposedAccuracyLoad.py
- Table A9: ProposedAccuracyCV/ProposedAccuracyCVLoad.py

## Software Setup

### Python dependencies
```
python 3.12.4
numpy 2.0
matplotlib 3.9.1
pandas 2.2.2
scipy 1.14
scikit-learn 1.5.1
tensorflow 2.11.0
```

## Data
This folder includes the retinal fundus image data and the MNIST data. The data for the other examples are not included here due to space limitations. Additionally, scripts and data results to generate Figures 2 and A5 as well as tables 1 and A3-A8 are not included due to space limitations.
