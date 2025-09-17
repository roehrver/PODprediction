"""
Data preprocessing module for risk evaluation.
Handles undersampling, outlier detection, and data splitting.
Refactored by Claude Opus 4.1 for improved readability and performance, then adjusted to work with the framework."""
module DataPrep

export random_undersampling,
       riemannian_distance_outlier_filter,
       create_patient_stratified_splits,
       threshold_probability,
       create_key_interactions

using StatsBase
using Statistics
using LinearAlgebra
using PosDefManifold
using PosDefManifoldML
using PyCall
using DataFrames

# Import sklearn components
sklearn_model_selection = pyimport("sklearn.model_selection")
StratifiedKFold = sklearn_model_selection.StratifiedKFold

function __init__()
    copy!(sklearn_model_selection, pyimport("sklearn.model_selection"))
end

"""
    create_key_interactions(df::DataFrame, key_columns::Vector{String}; include_original::Bool=true)

Create interaction features between key columns and all other columns.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `df`: Input DataFrame
- `key_columns`: List of columns to create interactions with
- `include_original`: Whether to include original features (default: true)

# Returns
- DataFrame with original features and interaction terms
"""
function create_key_interactions(df::DataFrame, key_columns::Vector{String}; include_original::Bool=true)
    # Start with original features if requested
    result = include_original ? copy(df) : DataFrame()
    
    # For each key column, create interactions with all other columns
    for key_col in key_columns
        other_cols = [col for col in names(df) if col != key_col]
        
        for other_col in other_cols
            # Create interaction name
            interaction_name = "$(key_col)_x_$(other_col)"
            
            # Skip if this interaction already exists
            if interaction_name in names(result)
                continue
            end
            
            # Create the interaction feature
            result[!, interaction_name] = df[!, key_col] .* df[!, other_col]
        end
    end
    
    return result
end

"""
    random_undersampling(train_classes, all_train_data, num_timeframes_train; kwargs...)

Randomly undersample the majority class (class 0) to balance the dataset.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `train_classes`: Vector of class labels for training patients
- `all_train_data`: Training data (Vector of Vectors of either Hermitian or Matrix{Float64})
- `num_timeframes_train`: Number of timeframes per training patient

# Keyword Arguments
- `medication_types`: Medication information for stratification (default: [])
- `mean_per_patient`: Geometric mean per patient (only for covariance data)
- `undersample_ratio`: Ratio of majority class to keep (default: 0.3)
- `n_splits`: Number of splits for stratified sampling (default: 3)
- `is_hermitian`: Whether data contains Hermitian matrices (default: true)

# Returns
- Tuple of (undersampled_train_data, undersampled_classes, updated_timeframes, mean_per_patient)
"""
function random_undersampling(
    train_classes::Vector,
    all_train_data::Vector,
    num_timeframes_train::Vector;
    medication_types::Vector = [],
    undersample_ratio::Float64 = 0.3,
    n_splits::Int = 3,
    is_hermitian::Bool = true
)
    # Convert to appropriate types
    train_classes_copy = Vector(train_classes)
    
    if is_hermitian
        train_data = Vector{Vector{Hermitian}}(all_train_data)
    else
        train_data = Vector{Vector{Matrix{Float64}}}(all_train_data)
    end
    
    # Find indices of majority class (class 0)
    indices_class_0 = findall(iszero, train_classes)
    
    # Determine indices to remove
    indices_to_remove = Int[]
    
    if length(unique(medication_types)) > 2
        # Stratified undersampling by medication
        println("Using stratified undersampling with $(n_splits) splits")
        
        # Create stratified splits
        stratified_kfold = sklearn_model_selection.StratifiedKFold(
            n_splits = n_splits,
            shuffle = true
        )
        
        test_indices = []
        for (train_idx, test_idx) in stratified_kfold.split(
            indices_class_0 .- 1,  # Convert to 0-based indexing for Python
            medication_types[indices_class_0]
        )
            push!(test_indices, test_idx .+ 1)  # Convert back to 1-based
        end
        
        indices_to_remove = indices_class_0[test_indices[1]]
    else
        # Random undersampling
        n_to_remove = Int(round(undersample_ratio * length(indices_class_0)))
        sampled_indices = zeros(Int, n_to_remove)
        StatsBase.knuths_sample!(indices_class_0, sampled_indices)
        indices_to_remove = sampled_indices
    end
    
    # Remove selected indices
    indices_to_remove = unique!(sort!(indices_to_remove))
    println("Removing $(length(indices_to_remove)) samples from majority class")
    
    deleteat!(train_data, indices_to_remove)
    deleteat!(train_classes_copy, indices_to_remove)
    
    current_timeframes = Vector(num_timeframes_train)
    deleteat!(current_timeframes, indices_to_remove)

    return train_data, train_classes_copy, current_timeframes, []
end

"""
    create_patient_stratified_splits(labels::Vector{Int}, samples_per_patient::Vector{Int}, n_folds::Int=5)

Create cross-validation splits stratified by patient labels, ensuring patients (not individual samples) 
are distributed across folds while maintaining label distribution.
Refactored by Claude Opus 4.1 for improved readability and performance.

This function performs patient-level stratification, which is critical in medical/clinical machine learning
to prevent data leakage where samples from the same patient appear in both training and test sets.

# Arguments
- `labels::Vector{Int}`: Patient-level labels/classes (one per patient)
- `samples_per_patient::Vector{Int}`: Number of samples for each patient (must match length of labels)
- `n_folds::Int=5`: Number of cross-validation folds (default: 5)

# Returns
- `Vector{Tuple{Vector{Int}, Vector{Int}}}`: Array of (train_indices, test_indices) tuples, where
  indices refer to sample positions in the flattened dataset
"""
function create_patient_stratified_splits(labels::Vector{Int}, samples_per_patient::Vector{Int}, n_folds::Int=5)
    n_patients = length(labels)
    patient_indices = 1:n_patients
    
    kf = StratifiedKFold(n_splits=n_folds, shuffle=true)
    kf.get_n_splits(patient_indices, labels)
    
    splits = []
    cumsum_samples = cumsum([0; samples_per_patient])
    
    for (train_patients, test_patients) in kf.split(patient_indices, labels)
        train_samples = Int[]
        test_samples = Int[]
        
        for patient_idx in train_patients
            start_idx = cumsum_samples[patient_idx + 1]
            end_idx = cumsum_samples[patient_idx + 2] - 1
            append!(train_samples, start_idx:end_idx)
        end
        
        for patient_idx in test_patients
            start_idx = cumsum_samples[patient_idx + 1]
            end_idx = cumsum_samples[patient_idx + 2] - 1
            append!(test_samples, start_idx:end_idx)
        end
        
        push!(splits, (train_samples, test_samples))
    end
    
    return splits
end

end