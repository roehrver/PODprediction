# POD Prediction

A Julia-based pipeline for predicting Postoperative Delirium (POD) using intraoperative EEG data analysis.

## Overview

Postoperative delirium (POD) is a common complication in elderly surgical patients, causing prolonged hospitalization, cognitive decline, and increased institutionalization. This repository implements a machine learning pipeline that uses routine intraoperative EEG monitoring to detect underlying brain vulnerability patterns predictive of POD beyond clinical risk factors.

The analysis was conducted on two cohorts of elderly patients from Charité Berlin: SuDoCo (n=1032, ≥ 60 years) and ePOD (n=263, ≥ 70 years), recorded 10 years apart with different anesthetic protocols. Our approach achieved balanced accuracy of 0.691 (AUC: 0.759) on SuDoCo, with robust transfer to the demographically distinct ePOD cohort.

## Repository Structure

### EEG Data Preprocessing Pipeline
- **ArtifactFiltering.jl** - Artifact detection and removal algorithms
- **Segment.jl** - EEG segmentation into analysis windows  
- **PreProcessing.jl** - General preprocessing functions (includes novel two-step burst suppression detection)


### Classification Pipeline
- **DataPrep.jl** - Data preparation and loading utilities
- **FeatureExtraction.jl** - Extraction of EEG features (burst suppression duration, power spectral densities, signal covariances)
- **Classification.jl** - Machine learning models and meta-classifier implementation
- **Utility.jl** - Helper functions and utilities
- **spectrum_groupedSHAP/** - SHAP analysis for feature interpretability
- - **Test_withRandom.jl** - Example using train and test set and randomly generated data

### Template Pipelines
- **PreprocessingPipeline.jl** - Main preprocessing workflow coordination (template - not complete pipeline)
- **PrototypicalCVSpectrumPipeline.jl** - Example cross-validation pipeline showing fold structure and workflow (template - not complete pipeline)


## Key Features

- **Novel burst suppression detection**: Two-step algorithm for robust burst suppression identification
- **Multi-feature approach**: Integration of three complementary EEG feature sets
- **Medication-specific modeling**: Eliminates pharmacological confounds by training anesthetic-specific models
- **Domain adaptation**: Robust transfer between different patient cohorts and recording protocols
- **Meta-classifier**: Integrates multiple approaches for optimal performance

## Method Highlights

- Addresses the challenge that maintenance anesthetic choice creates profound EEG differences that classifiers initially exploit as POD proxies
- Demonstrates that EEG-based vulnerability markers add crucial discriminative power when demographic predictors fail
- Particularly effective in homogeneous elderly populations where age and clinical scores lack sufficient resolution

## Dependencies

The pipeline uses the following Julia packages:
- DataFrames, CSV - Data manipulation
- StatsBase, Statistics, LinearAlgebra - Statistical computations
- FourierAnalysis, PosDefManifold, PosDefManifoldML, DSP - Signal processing and manifold operations
- PyCall - Integration with Python scikit-learn
- JSON, JLD - Data serialization

## Citations

This pipeline uses functions and methods based on the following papers, which should be cited if you use this code:

**EEG Manifold Operations and Covariance Analysis:**
- Congedo, M., & Jain, S. (2019). A Julia Package for manipulating Brain-Computer Interface Data in the Manifold of Positive Definite Matrices. *2019 IEEE International Conference on Systems, Man and Cybernetics (SMC)*, Bari, Italy. DOI: 10.1109/SMC.2019.8914223

**Synchronization Measures:**
- Congedo, M. (2018). Non-Parametric Synchronization Measures used in EEG and MEG. *Technical Report*. Available: https://hal.archives-ouvertes.fr/hal-01868538v2/document

**Machine Learning Framework:**
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.


## Note

The datasets (SuDoCo and ePOD) are not publicly available. This repository provides the analysis framework and template pipelines for researchers with access to similar EEG data collected during anesthesia.
