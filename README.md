# Machine-Learning-Guided Screening of Passivators for Perovskite via Boundary-Aware Weighting and Ensemble Learning
This repository contains the data and Python scripts used for the machine-learning-guided screening of molecular passivators for perovskite materials.

The workflow integrates literature-derived experimental data, molecular/electronic features, boundary-aware sample weighting, ensemble learning, model interpretation, candidate-property prediction, and auxiliary CAS-number checking. The repository is intended to support reproducibility of the analyses described in the associated manuscript.
## Data
The repository includes the raw data collected from the literature and used as the input dataset for model development and screening.

The raw data are provided to facilitate:

- inspection of the literature-derived dataset;
- reproduction of data preprocessing and model development;
- verification of molecular and experimental information used in the study;
- further development of machine-learning models for passivator screening.
## Python Scripts
#### `main.py`

Main script for the machine-learning workflow.

The script is used for model development and evaluation, including the processing of the input data and the machine-learning analysis described in the manuscript.

#### `shap.py`

Script for SHAP-based model interpretation.

It is used to analyze the contribution of molecular and electronic features to the model predictions and to generate SHAP-based feature-importance and dependence analyses.

#### `check_cas.py`

Script for checking CAS-related information for candidate molecules.

It is used as an step during candidate screening and information verification.

#### `predict.py`

Script for prediction of candidate molecules using the trained machine-learning workflow.

It is intended for applying the developed model to candidate compounds during virtual screening.
