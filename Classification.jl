"""
Classification module for risk evaluation.
Implements various classifiers for POD prediction.
Refactored by Claude Opus 4.1 for improved readability and performance, then adjusted to work with the framework."""
module Classification

export balanced_random_forest_prediction,
       covariance_prediction,
       spectrum_prediction

using PyCall
using Statistics
using DataFrames
using JSON
using LinearAlgebra
using PosDefManifold
using PosDefManifoldML

np = pyimport("numpy")  # Should be fast
sklearn = pyimport("sklearn")  # Should be fast  
println("sklearn...")
@pyimport joblib

println("joblib...")

# Replace @sk_import with direct PyCall imports
sklearn_metrics = pyimport("sklearn.metrics")
sklearn_svm = pyimport("sklearn.svm") 
sklearn_preprocessing = pyimport("sklearn.preprocessing")
sklearn_model_selection = pyimport("sklearn.model_selection")
sklearn_pipeline = pyimport("sklearn.pipeline")
sklearn_decomposition = pyimport("sklearn.decomposition")
imb_ensemble = pyimport("imblearn.ensemble")

# Access functions directly
roc_auc_score = sklearn_metrics.roc_auc_score
roc_curve = sklearn_metrics.roc_curve
balanced_accuracy_score = sklearn_metrics.balanced_accuracy_score
SVC = sklearn_svm.SVC
StandardScaler = sklearn_preprocessing.StandardScaler
GridSearchCV = sklearn_model_selection.GridSearchCV
StratifiedKFold = sklearn_model_selection.StratifiedKFold
make_pipeline = sklearn_pipeline.make_pipeline
Pipeline = sklearn_pipeline.Pipeline
PCA = sklearn_decomposition.PCA
BalancedRandomForestClassifier = imb_ensemble.BalancedRandomForestClassifier

"""
    balanced_random_forest_prediction(train_df, train_classes, test_df; kwargs...)

Train and predict using Balanced Random Forest Classifier from imbalanced-learn.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `train_df`: Training features DataFrame
- `train_classes`: Training class labels
- `test_df`: Test features DataFrame

# Keyword Arguments
- `key_columns`: Columns for creating interaction features (default: medication columns)
- `max_depth`: Maximum tree depth (default: 6)
- `n_estimators`: Number of trees (default: 200)

# Returns
- (test_predictions, train_predictions)
"""
function balanced_random_forest_prediction(
    train_df::DataFrame,
    train_classes::Vector,
    test_df::DataFrame;
    external_data::DataFrame = DataFrame(),
    key_columns::Vector{String} = ["aufDesflurane", "aufSevoflurane", "aufPropofol"],
    max_depth::Int = 6,
    n_estimators::Int = 200
)
    # Create interaction features
    train_features =  Array(create_key_interactions(train_df, key_columns))
    test_features = Array(create_key_interactions(test_df, key_columns))
    if nrow(external_data) > 0
        external_features = Array(create_key_interactions(external_data, key_columns))
    end

    # Convert to Array
    test_features = Array(test_features)
    if nrow(external_data) > 0
        external_features = Array(external_features)
    end

    # Initialize classifier
    classifier = BalancedRandomForestClassifier(
        max_depth = max_depth,
        n_estimators = n_estimators,
        bootstrap = true,
        random_state = 0,
        n_jobs = -1
    )
    
    # Train model
    classifier.fit(train_features, train_classes)
    
    # Make predictions (get probability of class 1)
    test_pred = classifier.predict_proba(test_features)[:, 2]
    train_pred = classifier.predict_proba(train_features)[:, 2]
    external_pred = Float64[]
    if nrow(external_data) > 0
        external_pred = classifier.predict_proba(external_features)[:, 2]
    end

    return test_pred, train_pred, external_pred
end

"""
    covariance_svm_prediction(train_data, test_data, train_classes, external_data; kwargs...)

SVM classification for covariance time series data.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments  
- `train_data`: Training covariance matrices
- `test_data`: Test covariance matrices
- `train_classes`: Training class labels
- `external_data`: External validation dataset

# Keyword Arguments
- `external_mean`: Geometric mean for external dataset
- `num_timeframes_test`: Timeframes per test patient
- `num_timeframes_train`: Timeframes per train patient
- `n_bags`: Number of ensemble models (default: 10)
- `n_samples_majority`: Samples for minority class (default: 15)
- `undersampling_params`: [ratio, n_splits] (default: [0.2, 4])
- `save_models`: Whether to save models (default: false)
- `model_folder`: Folder for saving models
- `fold_number`: Current fold number for cross-validation

# Returns
- Tuple of (test_predictions, train_predictions, external_predictions)
"""
function covariance_prediction(
    train_data::Vector,
    test_data::Vector,
    train_classes::Vector,
    external_data::Vector,
    external_mean::Union{Hermitian, Nothing};
    medication_train::Vector = [],
    num_timeframes_test::Vector,
    num_timeframes_train::Vector,
    mean_per_patient_test::Vector{Hermitian} = Hermitian[],
    n_bags::Int = 10,
    n_samples_majority::Int = 15,
    undersampling_params::Vector = [0.2, 4]
)
    # Initialize prediction arrays
    n_train_samples = sum(num_timeframes_train)
    n_test_samples = sum(num_timeframes_test)
    n_external_samples = sum(length.(external_data))
    
    train_predictions = zeros(n_train_samples)
    test_predictions = zeros(n_test_samples)
    external_predictions = zeros(n_external_samples)

    # Training loop
    models_trained = 0
    rejection_count = 0
    
    while models_trained < n_bags
        # Undersample training data
        undersampled_train, undersampled_classes, current_timeframes, current_means = 
            random_undersampling(
                train_classes,
                train_data,
                num_timeframes_train;
                medication_types = medication_train,
                undersample_ratio = undersampling_params[1],
                n_splits = Int(undersampling_params[2])
            )
        
        # Sample timeframes for each patient
        #n_pos = sum(undersampled_classes)
        #n_neg = length(undersampled_classes) - n_pos
        ratio = 2#n_neg / n_pos
        
        samples_per_class = [n_samples_majority, Int(ceil(n_samples_majority * ratio))]
        
        # Create training batch
        train_batch = Hermitian[]
        train_batch_labels = Int[]
        
        for i in 1:length(undersampled_classes)
            n_samples = samples_per_class[undersampled_classes[i] + 1]
            for j in 1:n_samples
                idx = rand(1:current_timeframes[i])
                push!(train_batch, Hermitian(undersampled_train[i][idx]))
                push!(train_batch_labels, undersampled_classes[i])
            end
        end
        
        # Compute geometric mean for normalization
        class_0_mean = mean(logEuclidean,
            ℍVector([train_batch[i] for i in 1:length(train_batch)
                    if train_batch_labels[i] == 0])
        )
        class_1_mean = mean(logEuclidean,
            ℍVector([train_batch[i] for i in 1:length(train_batch)
                    if train_batch_labels[i] == 1])
        )
        
        reference_mean = mean(logEuclidean,
            ℍVector([Hermitian(class_0_mean), Hermitian(class_1_mean)])
        )
        
        # Extract features
        train_features = Array(extract_log_euclidean_features(
            ℍVector(train_batch);
            reference_mean = reference_mean,
            compute_mean = false
        ))

        # Prepare all data for evaluation
        all_train_flat = [Hermitian(train_data[i][j]) 
                         for i in 1:length(train_data) 
                         for j in 1:num_timeframes_train[i]]
        all_train_features = extract_log_euclidean_features(
            ℍVector(all_train_flat);
            reference_mean = reference_mean,
            compute_mean = false
        )
        
        test_flat = [Hermitian(test_data[i][j])
                    for i in 1:length(test_data)
                    for j in 1:num_timeframes_test[i]]
        test_features = Array(extract_log_euclidean_features(
            ℍVector(test_flat);
            reference_mean = reference_mean,
            compute_mean = false
        ))
        
        external_flat = [Hermitian(external_data[i][j])
                        for i in 1:length(external_data)
                        for j in 1:length(external_data[i])]
        external_features = extract_log_euclidean_features(
            ℍVector(external_flat);
            reference_mean = isnothing(external_mean) ? reference_mean : external_mean,
            compute_mean = false
        )
        
        # Create CV splits based on patients
        samples_per_patient = [samples_per_class[label + 1] for label in undersampled_classes]
        cv_splits = create_patient_stratified_splits(undersampled_classes, samples_per_patient)        
        # Setup SVM pipeline
        svm_pipeline = sklearn_pipeline.make_pipeline(
            sklearn_svm.SVC(
                class_weight = "balanced",
                random_state = 0,
                decision_function_shape = "ovo",
                probability = false
            )
        )
        
        # Grid search for hyperparameters
        param_grid = Dict(
            "svc__C" => [0.25, 1, 5],
            "svc__kernel" => ["linear"]
        )
        
        grid_search = sklearn_model_selection.GridSearchCV(
            svm_pipeline,
            param_grid,
            scoring = "roc_auc",
            n_jobs = -1,
            refit = true,
            cv = cv_splits
        )
        println(size(train_features))
        println(length(train_batch_labels))
        # Fit model
        grid_search.fit(train_features, train_batch_labels)
        best_model = grid_search.best_estimator_
        
        println("Best params: ", grid_search.best_params_)
        println("Best score: ", grid_search.best_score_)
        
        # Evaluate on full training set
        train_pred_temp = best_model.predict(all_train_features)
        
        # Calculate per-patient accuracy
        cumsum_timeframes = cumsum([0; num_timeframes_train])
        patient_predictions = [
            mean(train_pred_temp[cumsum_timeframes[j]+1:cumsum_timeframes[j+1]]) >= 0.5 ? 1 : 0
            for j in 1:length(train_classes)
        ]
        
        train_auc = sklearn_metrics.roc_auc_score(train_classes, patient_predictions)
        
        # Accumulate predictions
        train_predictions .+= best_model.predict(all_train_features)
        test_predictions .+= best_model.predict(test_features)
        external_predictions .+= best_model.predict(external_features)
        
        models_trained += 1
    end
    
    # Average predictions over all models
    train_predictions ./= n_bags
    test_predictions ./= n_bags
    external_predictions ./= n_bags
    
    # Convert to per-patient predictions
    train_patient_pred = aggregate_patient_predictions(train_predictions, num_timeframes_train)
    test_patient_pred = aggregate_patient_predictions(test_predictions, num_timeframes_test)
    external_patient_pred = aggregate_patient_predictions(external_predictions, 
                                                       [length(d) for d in external_data])
    
    return test_patient_pred, train_patient_pred, external_patient_pred
end


# Helper functions

function aggregate_patient_predictions(predictions::Vector, timeframes_per_patient::Vector)
    """Aggregate timeframe predictions to patient level."""
    cumsum_timeframes = cumsum([0; timeframes_per_patient])
    patient_predictions = Float64[]
    
    for i in 1:length(timeframes_per_patient)
        start_idx = cumsum_timeframes[i] + 1
        end_idx = cumsum_timeframes[i + 1]
        push!(patient_predictions, mean(predictions[start_idx:end_idx]))
    end
    
    return patient_predictions
end


# ===== Model Training Functions =====

"""
    train_svm_ensemble_member(train_features::Vector, 
                            train_labels::Vector{Int},
                            cv_splits::Vector)

Train a single SVM model with grid search cross-validation.
"""
function train_svm_ensemble_member(train_features::Vector, 
                                  train_labels::Vector{Int},
                                  cv_splits::Vector)
    pipeline = make_pipeline(
        SVC(
            class_weight="balanced",
            random_state=0,
            decision_function_shape="ovo"
        )
    )
    
    param_grid = Dict(
        "svc__C" => [0.5, 1, 5],
        "svc__kernel" => ["rbf"]
    )
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        refit=true,
        cv=cv_splits
    )
    
    grid_result = grid_search.fit(train_features, train_labels)
    
    println("Best parameters: ", grid_result.best_params_)
    println("Best score: ", grid_result.best_score_)
    
    return grid_result.best_estimator_
end


"""
    spectrum_prediction(train_data, test_data, train_labels, test_epod, baseline_mean, 
                       medication, medication_train, n_timeframes_test, n_timeframes_train;
                       n_bags=10, n_samples=15, spectrum_length=46, 
                       undersampling_params=[0.2, 4], fold=1, save_results=false,
                       model_folder="models", filename="All", output_folder="explanations")

Perform spectrum-based prediction using an ensemble of SVM models.

# Arguments
- `train_data`: Training data for each patient
- `test_data`: Test data for each patient  
- `train_labels`: Binary labels for training patients
- `test_epod`: EPOD test data
- `baseline_mean`: Baseline mean for normalization
- `medication`: Medication information
- `medication_train`: Training medication information
- `n_timeframes_test`: Number of timeframes per test patient
- `n_timeframes_train`: Number of timeframes per training patient
- `n_bags`: Number of ensemble members
- `n_samples`: Samples per class per patient
- `spectrum_length`: Length of spectrum features
- `undersampling_params`: Parameters for undersampling [ratio, split]
- `fold`: Current fold number
- `save_results`: Whether to save results
- `model_folder`: Folder for saving models
- `filename`: Base filename for outputs
- `output_folder`: Folder for saving explanations

# Returns
- `test_predictions`: Patient-level test predictions
- `train_predictions`: Patient-level training predictions  
- `epod_predictions`: EPOD predictions
"""
function spectrum_prediction(
    train_data,
    test_data,
    train_labels,
    test_epod,
    baseline_mean,
    medication,
    medication_train,
    n_timeframes_test,
    n_timeframes_train;
    n_bags=10,
    n_samples=15,
    spectrum_length=46,
    undersampling_params=[0.2, 4],
    fold=1,
    save_results=false,
    model_folder=homedir() * "/Promotion/BCI2/models",
    filename="All",
    output_folder=homedir() * "/Promotion/BCI2/explanations"
)
    # Initialize predictions
    test_predictions_all = zeros(sum(n_timeframes_test))
    train_predictions_all = zeros(sum(n_timeframes_train))
    epod_predictions_all = zeros(sum(length.(test_epod)))
    
    # Prepare fixed samples for consistent SHAP analysis
    fixed_train_samples, fixed_train_labels = prepare_fixed_samples(
        train_data, train_labels, n_timeframes_train, spectrum_length, 2500
    )
    
    # Save fixed training data if requested
    if save_results
        shap_data = Dict(
            "fixed_train" => fixed_train_samples,
            "fixed_train_labels" => fixed_train_labels
        )
        save_shap_data(shap_data, output_folder, filename, fold)
    end
    
    models = []
    
    # Train ensemble
    for bag_idx in 1:n_bags
        # Perform undersampling
        sampled_train_data, sampled_labels, sampled_timeframes = random_undersampling(
            train_labels,
            Vector{Vector{Matrix{Float64}}}(train_data),
            n_timeframes_train;
            medication_types=medication_train,
            undersample_ratio=undersampling_params[1],
            n_splits=Int(undersampling_params[2]),
            is_hermitian=false
        )

        # Calculate samples per class
        n_positive = sum(sampled_labels)
        n_negative = length(sampled_labels) - n_positive
        samples_per_class = [n_samples, n_samples]
        
        # Extract training features for this bag
        bag_train_features = []
        bag_train_labels = Int[]
        
        for (patient_idx, label) in enumerate(sampled_labels)
            n_patient_samples = samples_per_class[label + 1]
            patient_features = extract_patient_features(
                sampled_train_data[patient_idx],
                sampled_timeframes[patient_idx],
                spectrum_length,
                n_patient_samples
            )
            append!(bag_train_features, patient_features)
            append!(bag_train_labels, fill(label, n_patient_samples))
        end
        
        # Add fixed samples to training set
        all_train_features = vcat(bag_train_features, fixed_train_samples)
        all_train_labels = vcat(bag_train_labels, fixed_train_labels)
        
        # Calculate training mean
        train_mean = mean(all_train_features)
        println("Training mean size: ", size(train_mean))
        
        # Prepare test features
        test_features = prepare_timeframe_features(test_data, n_timeframes_test, spectrum_length)
        
        # Prepare all training features for evaluation
        all_train_eval_features = prepare_timeframe_features(
            Vector{Vector{Matrix{Float64}}}(train_data),
            n_timeframes_train,
            spectrum_length
        )
        
        # Prepare EPOD features with baseline adjustment
        epod_features = []
        for (patient_idx, patient_data) in enumerate(test_epod)
            for timeframe_idx in 1:length(patient_data)
                feature = extract_spectrum_features(patient_data[timeframe_idx], spectrum_length)
                if baseline_mean !== nothing
                    adjusted_feature = feature .- (baseline_mean .- train_mean)
                else
                    adjusted_feature = feature
                end
                push!(epod_features, adjusted_feature)
            end
        end
        println("EPOD features size: ", size(epod_features))

        println("EPOD features size: ", size(epod_features[1]))

        println("All train features size: ", size(all_train_features))
        
        # Create CV splits
        samples_per_patient = vcat(
            [samples_per_class[label + 1] for label in sampled_labels],
            ones(Int, length(fixed_train_samples))
        )
        cv_splits = create_patient_stratified_splits(
            vcat(sampled_labels, fixed_train_labels),
            samples_per_patient
        )
        
        # Train model
        model = train_svm_ensemble_member(all_train_features, all_train_labels, cv_splits)
        push!(models, model)
        
        # Save model if requested
        if save_results
            save_models([model], model_folder, "$(filename)_bag$(bag_idx)", fold)
        end
        
        # Make predictions
        train_predictions_all .+= model.predict(all_train_eval_features)
        test_predictions_all .+= model.predict(test_features)
        epod_predictions_all .+= model.predict(epod_features)
        
        # Save test samples for this bag if requested
        if save_results && bag_idx == 1  # Save only for first bag as example
            test_samples_bag, test_labels_bag = prepare_fixed_samples(
                test_data, train_labels, n_timeframes_test, spectrum_length, 500
            )
            
            shap_data = Dict(
                "test_samples_bag$(bag_idx)" => test_samples_bag,
                "test_labels_bag$(bag_idx)" => test_labels_bag
            )
            save_shap_data(shap_data, output_folder, filename, fold)
        end
    end
    
    # Average ensemble predictions
    train_predictions_all ./= n_bags
    test_predictions_all ./= n_bags
    epod_predictions_all ./= n_bags
    
    # Aggregate to patient level
    train_means = aggregate_patient_predictions(
        train_predictions_all, n_timeframes_train
    )
    test_means = aggregate_patient_predictions(
        test_predictions_all, n_timeframes_test
    )
    
    # Aggregate EPOD predictions
    epod_timeframes = [length(patient_data) for patient_data in test_epod]
    epod_means = aggregate_patient_predictions(epod_predictions_all, epod_timeframes)
    
    # Save metadata if requested
    if save_results && !isempty(models)
        metadata = Dict(
            "n_models" => n_bags,
            "n_features" => spectrum_length,
            "n_fixed_train_samples" => length(fixed_train_samples),
            "feature_type" => "spectrum"
        )
        save_metadata(metadata, output_folder, filename, fold)
    end
    
    return test_means, train_means, epod_means
end


# ===== Save/Load Functions =====
"""
    save_shap_data(data::Dict, folder_path::String, filename::String, fold::Int)

Save data for SHAP analysis in numpy format.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: Dictionary containing data to save
- `folder_path`: Base folder path for saving
- `filename`: Base filename for the saved files
- `fold`: Fold number for cross-validation

# Returns
- Nothing (saves files to disk)
"""
function save_shap_data(data::Dict, folder_path::String, filename::String, fold::Int)
    shap_folder = joinpath(folder_path, "shap_data")
    mkpath(shap_folder)
    
    np = pyimport("numpy")
    
    for (key, value) in data
        if isa(value, Vector) && !isempty(value) && isa(value[1], Vector)
            # Convert vector of vectors to matrix
            matrix = reduce(vcat, [reshape(x, 1, :) for x in value])
            filepath = joinpath(shap_folder, "$(key)_spectrum_fold$(fold)_$(filename).npy")
            np.save(filepath, matrix)
        elseif isa(value, Vector)
            # Save vector directly
            filepath = joinpath(shap_folder, "$(key)_spectrum_fold$(fold)_$(filename).npy")
            np.save(filepath, value)
        end
    end
    
    println("SHAP data saved to: $shap_folder")
end

"""
    save_models(models::Vector, folder_path::String, filename::String, fold::Int)

Save trained models using joblib.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `models`: Vector of trained models to save
- `folder_path`: Folder path for saving models
- `filename`: Base filename for the saved models
- `fold`: Fold number for cross-validation

# Returns
- Nothing (saves models to disk)
"""
function save_models(models::Vector, folder_path::String, filename::String, fold::Int)
    mkpath(folder_path)
    
    for (idx, model) in enumerate(models)
        model_path = joinpath(folder_path, "pipeline_spectrum_fold$(fold)_$(filename)_bag$(idx).pkl")
        joblib.dump([model], model_path)
    end
    
    println("Models saved to: $folder_path")
end

"""
    save_metadata(metadata::Dict, folder_path::String, filename::String, fold::Int)

Save metadata as JSON.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `metadata`: Dictionary containing metadata to save
- `folder_path`: Base folder path for saving
- `filename`: Base filename for the metadata file
- `fold`: Fold number for cross-validation

# Returns
- Nothing (saves metadata to disk)
"""
function save_metadata(metadata::Dict, folder_path::String, filename::String, fold::Int)
    shap_folder = joinpath(folder_path, "shap_data")
    mkpath(shap_folder)
    
    filepath = joinpath(shap_folder, "metadata_spectrum_fold$(fold)_$(filename).json")
    open(filepath, "w") do f
        JSON.print(f, metadata)
    end
end



# Include helper functions from parent modules
using ..DataPrep: random_undersampling, create_patient_stratified_splits, create_key_interactions
using ..FeatureExtraction: extract_log_euclidean_features, extract_spectrum_features, extract_patient_features, prepare_fixed_samples, prepare_timeframe_features

end