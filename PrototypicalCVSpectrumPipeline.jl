using PyCall
using DataFrames, CSV
using StatsBase, Statistics, LinearAlgebra, Random
using FourierAnalysis, PosDefManifold, PosDefManifoldML
using JSON, JLD


include(homedir() * "path/to//DataPrep.jl")
include(homedir() * "path/to/FeatureExtraction.jl")
include(homedir() * "path/to/Classification.jl")

using .DataPrep
using .FeatureExtraction
using .Classification


# Import sklearn modules
sklearn_model_selection = pyimport("sklearn.model_selection")
sklearn_metrics = pyimport("sklearn.metrics")
sklearn_svm = pyimport("sklearn.svm")
sklearn_preprocessing = pyimport("sklearn.preprocessing")
sklearn_pipeline = pyimport("sklearn.pipeline")
sklearn_linear_model = pyimport("sklearn.linear_model")

# Create direct references to sklearn functions/classes
StratifiedKFold = sklearn_model_selection.StratifiedKFold
roc_auc_score = sklearn_metrics.roc_auc_score
roc_curve = sklearn_metrics.roc_curve
balanced_accuracy_score = sklearn_metrics.balanced_accuracy_score
SVC = sklearn_svm.SVC
StandardScaler = sklearn_preprocessing.StandardScaler
GridSearchCV = sklearn_model_selection.GridSearchCV
RepeatedStratifiedKFold = sklearn_model_selection.RepeatedStratifiedKFold
make_pipeline = sklearn_pipeline.make_pipeline
Pipeline = sklearn_pipeline.Pipeline
LogisticRegression = sklearn_linear_model.LogisticRegression
imb = pyimport("imblearn")

# Configuration parameters
const timeframes = 10
const spectruml = 46
const spectrum_length = 46

# ML parameters (szenario4 values)
const bags = 10
const noS = 20
const undersampling = [0.2, 4]

anesthetic = "propofol"  # "propofol", "desflurane", "sevoflurane"

# =============================================================================
# DATA LOADING SECTION - PLACEHOLDER
# =============================================================================
# TODO: Load and preprocess the following data structures:
# - sudoco_df: DataFrame with patient metadata and outcomes
# - RefFiltered: Hermitian matrices for EEG covariance data
# - RefSpectrum: Spectral data arrays
# - df_epod: EPOD dataset DataFrame
# - RefSpectrum_EPOD: EPOD spectral data
# - exclude: DataFrame with files to exclude
# - exclude_epod: DataFrame with EPOD files to exclude
#
# Required preprocessing steps:
# 1. Filter by anesthetic type (propofol, desflurane, sevoflurane)
# 2. Exclude problematic files based on exclude lists
# 3. Remove cases with insufficient timeframes (<5)
# 4. Reduce spectrum length to spectruml
# 5. Calculate mean spectra for EPOD data
#
# Required index definitions:
# - propofolIndex: indices of propofol patients in sudoco_df dataset
# - desfluraneIndex: indices of desflurane patients in sudoco_df dataset  
# - sevofluraneIndex: indices of sevoflurane patients in sudoco_df dataset
# - propofolIndex_EPOD: indices of propofol patients in EPOD dataset
# - desfluraneIndex_EPOD: indices of desflurane patients in EPOD dataset
# - sevofluraneIndex_EPOD: indices of sevoflurane patients in EPOD dataset
# =============================================================================

# Placeholder variables - replace with actual data loading
global sudoco_df, RefFiltered, RefSpectrum, df_epod, RefSpectrum_EPOD
global propofolIndex, desfluraneIndex, sevofluraneIndex
global propofolIndex_EPOD, desfluraneIndex_EPOD, sevofluraneIndex_EPOD
global meanperPatient

# Cross-validation setup

# Generate CV folds based on anesthetic type
indextestfolds = []
indextrainfolds = []


println("setting up folds....")
parted = 5
kf = RepeatedStratifiedKFold(n_splits=parted, n_repeats=10, random_state=42)

indextestfolds=[]
indextrainfolds=[]
if anesthetic=="propofol"|| anesthetic=="all"
	println(anesthetic)
    for (train_index, test_index) in kf.split(propofolIndex, df[!,:POD_diagnosis][propofolIndex])
        append!(indextestfolds, [propofolIndex[test_index.+1]])
        append!(indextrainfolds, [propofolIndex[train_index.+1]])
    end
end

if anesthetic=="desflurane"|| anesthetic=="volatile"
	println(anesthetic)
    for (train_index, test_index) in kf.split(desfluraneIndex, df[!,:POD_diagnosis][desfluraneIndex])
        append!(indextestfolds, [desfluraneIndex[test_index.+1]])
        append!(indextrainfolds, [desfluraneIndex[train_index.+1]])
    end
end
if anesthetic=="sevoflurane"
	println(anesthetic)
    for  (i, (train_index, test_index)) in enumerate(kf.split(sevofluraneIndex, df[!,:POD_diagnosis][sevofluraneIndex]))
        append!(indextestfolds, [sevofluraneIndex[test_index.+1]])
        append!(indextrainfolds, [sevofluraneIndex[train_index.+1]])
    end
end

if anesthetic=="all"
	println(anesthetic)	
for  (i, (train_index, test_index)) in enumerate(kf.split(desfluraneIndex, df[!,:POD_diagnosis][desfluraneIndex]))
    append!(indextestfolds[i], desfluraneIndex[test_index.+1])
    append!(indextrainfolds[i], desfluraneIndex[train_index.+1])
end
for  (i, (train_index, test_index)) in enumerate(kf.split(sevofluraneIndex, df[!,:POD_diagnosis][sevofluraneIndex]))
    append!(indextestfolds[i], sevofluraneIndex[test_index.+1])
    append!(indextrainfolds[i], sevofluraneIndex[train_index.+1])
end
end

if anesthetic=="volatile"
    for  (i, (train_index, test_index)) in enumerate(kf.split(sevofluraneIndex, df[!,:POD_diagnosis][sevofluraneIndex]))
    append!(indextestfolds[i], sevofluraneIndex[test_index.+1])
    append!(indextrainfolds[i], sevofluraneIndex[train_index.+1])
end
end


println(size(indextestfolds[1]))
println(size(indextrainfolds[1]))

const propofolIndex_EPOD = df_epod |>
    x -> filter(:AM => ==(1), x)[!,"index"]

const desfluraneIndex_EPOD = df_epod |>
    x -> filter(:AM => ==(3), x)[!,"index"]

const sevofluraneIndex_EPOD = df_epod |>
    x -> filter(:AM => ==(2), x)[!,"index"]


#split epod into parted folds: training and test folds swapped (small training for domain adoption)
indextestfoldsEPOD = []
indextrainfoldsEPOD = []

if anesthetic=="propofol"|| anesthetic=="all"
    for (train_index, test_index) in kf.split(propofolIndex_EPOD, df_epod[!,:POD_diagnosis][propofolIndex_EPOD])
        append!(indextrainfoldsEPOD, [propofolIndex_EPOD[test_index.+1]])
        append!(indextestfoldsEPOD, [propofolIndex_EPOD[train_index.+1]])
    end
end
if anesthetic=="desflurane"|| anesthetic=="volatile"
    for (train_index, test_index) in kf.split(desfluraneIndex_EPOD, df_epod[!,:POD_diagnosis][desfluraneIndex_EPOD])
        append!(indextrainfoldsEPOD, [desfluraneIndex_EPOD[test_index.+1]])
        append!(indextestfoldsEPOD, [desfluraneIndex_EPOD[train_index.+1]])
    end
end
if anesthetic=="sevoflurane"
    for (train_index, test_index) in kf.split(sevofluraneIndex_EPOD, df_epod[!,:POD_diagnosis][sevofluraneIndex_EPOD])
        append!(indextrainfoldsEPOD, [sevofluraneIndex_EPOD[test_index.+1]])
        append!(indextestfoldsEPOD, [sevofluraneIndex_EPOD[train_index.+1]])
    end
end

if anesthetic=="all"
	println(anesthetic)	
for  (i, (train_index, test_index)) in enumerate(kf.split(desfluraneIndex_EPOD, df_epod[!,:POD_diagnosis][desfluraneIndex_EPOD]))
    append!(indextrainfoldsEPOD[i], desfluraneIndex_EPOD[test_index.+1])
    append!(indextestfoldsEPOD[i], desfluraneIndex_EPOD[train_index.+1])
end
for  (i, (train_index, test_index)) in enumerate(kf.split(sevofluraneIndex_EPOD, df_epod[!,:POD_diagnosis][sevofluraneIndex_EPOD]))
    append!(indextrainfoldsEPOD[i], sevofluraneIndex_EPOD[test_index.+1])
    append!(indextestfoldsEPOD[i], sevofluraneIndex_EPOD[train_index.+1])
end
end

if anesthetic=="volatile"
for  (i, (train_index, test_index)) in enumerate(kf.split(sevofluraneIndex_EPOD, df_epod[!,:POD_diagnosis][sevofluraneIndex_EPOD]))
    append!(indextrainfoldsEPOD[i], sevofluraneIndex_EPOD[test_index.+1])
    append!(indextestfoldsEPOD[i], sevofluraneIndex_EPOD[train_index.+1])
end
end

# Main cross-validation loop
repeat_results = Dict{Int, Dict{String, Vector}}()

for fold = 1:parted
    
    # Get fold indices
    indextest = indextestfolds[fold]
    indextrain = indextrainfolds[fold]
    indextestEPOD = indextestfoldsEPOD[fold]
    indextrainEPOD = indextrainfoldsEPOD[fold]
    
    # Calculate timeframes for each patient
    numberoftimeframesTest = Int.([minimum([length(RefSpectrum[indextest][i]), 
                                          length(RefFiltered[indextest][i])]) 
                                  for i = 1:size(indextest)[1]])
    numberoftimeframesTrain = Int.([minimum([length(RefSpectrum[indextrain][i]),
                                           length(RefFiltered[indextrain][i])]) 
                                   for i = 1:size(indextrain)[1]])
    
    # Prepare spectral data
    STest = [RefSpectrum[indextest][i][1:numberoftimeframesTest[i]] 
             for i = 1:size(indextest)[1]]
    STrain = [RefSpectrum[indextrain][i][1:numberoftimeframesTrain[i]] 
              for i = 1:size(indextrain)[1]]
    STest_EPOD = [RefSpectrum_EPOD[indextestEPOD[i]] 
                  for i = 1:size(indextestEPOD)[1]]
    STrain_EPOD = [RefSpectrum_EPOD[indextrainEPOD[i]] 
                   for i = 1:size(indextrainEPOD)[1]]
    
    # Get class labels
    testclassPP = sudoco_df[!, :POD_diagnosis][indextest]
    trainclassPP = sudoco_df[!, :POD_diagnosis][indextrain]
    testclassPP_EPOD = df_epod[!, :POD_diagnosis][indextestEPOD]
    trainclassPP_EPOD = df_epod[!, :POD_diagnosis][indextrainEPOD]
    
    # Determine medication types
    propofolindex = findall(in(propofolIndex), indextest)
    desfluraneindex = findall(in(desfluraneIndex), indextest)
    rest = findall(in(sevofluraneIndex), indextest)
    
    proptrain = findall(in(propofolIndex), indextrain)
    destrain = findall(in(desfluraneIndex), indextrain)
    resttrain = findall(in(sevofluraneIndex), indextrain)
    
    medication = [4 for i = 1:size(indextest)[1]]
    medicationT = [4 for i = 1:size(indextrain)[1]]
    
    medication[propofolindex] .= 1
    medication[desfluraneindex] .= 2
    medication[rest] .= 3
    medicationT[proptrain] .= 1
    medicationT[destrain] .= 2
    medicationT[resttrain] .= 3
    
    medicationEPOD = [4 for i = 1:size(indextestEPOD)[1]]
    medicationEPOD[findall(in(propofolIndex_EPOD), indextestEPOD)] .= 1
    medicationEPOD[findall(in(desfluraneIndex_EPOD), indextestEPOD)] .= 2
    medicationEPOD[findall(in(sevofluraneIndex_EPOD), indextestEPOD)] .= 3
    
    # Calculate baseline means for domain adaptation
    indextrainPODEPOD = indextrainEPOD[trainclassPP_EPOD .== 1]
    indextrainNoPOD = indextrainEPOD[trainclassPP_EPOD .== 0]
    
    meanNoPOD = mean(meanperPatient[indextrainNoPOD])
    meanPOD = mean(meanperPatient[indextrainPODEPOD])
    Bmean = (meanPOD + meanNoPOD) / 2
    
    # Run spectrum classification using function from Classification module
    pS, pST, pSEPOD = spectrum_prediction(
        STrain, STest, trainclassPP, STest_EPOD, Bmean,
        medication, medicationT, numberoftimeframesTest, numberoftimeframesTrain;
        n_bags=bags, n_samples=noS, spectrum_length=spectrum_length,
        undersampling_params=undersampling, fold=fold, save_results=false
    )
    
    # Store fold results
    repeat_results[fold] = Dict{String, Vector}(
        "train_probs" => pST,
        "test_probs" => pS,
        "test_probs_EPOD" => pSEPOD,
        "train_classes" => trainclassPP,
        "test_classes" => testclassPP,
        "test_classes_EPOD" => testclassPP_EPOD,
        "medication" => medication,
        "medication_EPOD" => medicationEPOD,
        "medication_train" => medicationT,
    )
    
    # =============================================================================
    # RESULTS SAVING SECTION - PLACEHOLDER
    # =============================================================================
    # TODO: Save results to files at the end of cross-validation:
    # 1. Save repeat_results as JSON
    # 2. Save results as JLD file
    # Use appropriate file naming convention
    # =============================================================================
end

println("Cross-validation completed!")