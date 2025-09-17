"""
Feature extraction module for EEG/BCI data.
Handles covariance and spectrum-based feature extraction.
Refactored by Claude Opus 4.1 for improved readability and performance, then adjusted to work with the framework.
"""
module FeatureExtraction

export extract_log_euclidean_features,
       extract_patient_features,
       extract_spectrum_features,
       prepare_fixed_samples,
       prepare_timeframe_features

using LinearAlgebra
using Statistics
using PosDefManifold
using PosDefManifoldML


"""
    extract_log_euclidean_features(covariances; kwargs...)

Extract features from covariance matrices using Log-Euclidean framework.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `covariances`: Vector of Hermitian covariance matrices

# Keyword Arguments
- `reference_mean`: Reference mean for tangent space projection (optional)
- `compute_mean`: Whether to compute the mean if not provided (default: true)
- `vec_range`: Range of channels to include (default: all)

# Returns
- If compute_mean: (feature_matrix, geometric_mean)
- Otherwise: feature_matrix
"""
function extract_log_euclidean_features(
    covariances::ℍVector;
    reference_mean::Union{ℍ, Nothing} = nothing,
    compute_mean::Bool = true,
    vec_range::UnitRange = 1:size(covariances[1], 2),
    transpose_output::Bool = true
)
    n_samples = length(covariances)
    n_channels = size(covariances[1], 1)
    
    # Compute or use provided mean
    should_compute_mean = isnothing(reference_mean)
    if should_compute_mean
        geometric_mean = mean(logEuclidean, covariances)
    else
        geometric_mean = reference_mean
    end
    
    # Compute feature dimension
    vec_dim = (length(vec_range) * (length(vec_range) + 1)) ÷ 2
    
    # Extract features
    if transpose_output
        features = Matrix{eltype(covariances[1])}(undef, n_samples, vec_dim)
        for i in 1:n_samples
            # Project to tangent space
            rlf = log(covariances[i]) - log(geometric_mean)
            features[i, :] = vecP(rlf; range = vec_range)
        end
    end
    return should_compute_mean ? (features, geometric_mean) : features
end


"""
    extract_spectrum_features(data::Matrix, spectrum_length::Int)

Extract spectrum features by averaging over dimensions.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: Input data matrix
- `spectrum_length`: Length of the spectrum

# Returns
- Vector of averaged spectrum features
"""
function extract_spectrum_features(data::Matrix, spectrum_length::Int)
    return vec(reshape(mean(data, dims=2), spectrum_length))
end

"""
    extract_patient_features(patient_data::Vector{Matrix{Float64}}, 
                            n_timeframes::Int,
                            spectrum_length::Int, 
                            n_samples::Int)

Extract random spectrum features from a patient's data.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `patient_data`: Vector of matrices containing patient's timeframe data
- `n_timeframes`: Number of available timeframes
- `spectrum_length`: Length of the spectrum
- `n_samples`: Number of random samples to extract

# Returns
- Vector of extracted features
"""
function extract_patient_features(patient_data::Vector{Matrix{Float64}}, 
                                 n_timeframes::Int,
                                 spectrum_length::Int, 
                                 n_samples::Int)
    features = []
    for _ in 1:n_samples
        random_timeframe = rand(1:n_timeframes)
        feature = extract_spectrum_features(patient_data[random_timeframe], spectrum_length)
        push!(features, feature)
    end
    return features
end

"""
    prepare_fixed_samples(data::Vector{Vector{Matrix{Float64}}}, 
                         labels::Vector{Int}, 
                         n_timeframes::Vector{Int},
                         spectrum_length::Int,
                         target_per_class::Int=2500)

Prepare fixed training/test samples for consistent SHAP analysis.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: Vector of patient data (each patient has vector of timeframe matrices)
- `labels`: Vector of patient labels (0/1)
- `n_timeframes`: Vector of timeframe counts per patient
- `spectrum_length`: Length of the spectrum
- `target_per_class`: Target number of samples per class (default: 2500)

# Returns
- Tuple of (fixed_samples, fixed_labels)
"""
function prepare_fixed_samples(data::Vector{Vector{Matrix{Float64}}}, 
                              labels::Vector{Int}, 
                              n_timeframes::Vector{Int},
                              spectrum_length::Int,
                              target_per_class::Int=2500)
    n_patients = length(labels)
    n_positive = sum(labels)
    n_negative = n_patients - n_positive
    
    samples_per_positive = Int(ceil(target_per_class / max(n_positive, 1)))
    samples_per_negative = Int(ceil(target_per_class / max(n_negative, 1)))
    
    fixed_samples = []
    fixed_labels = Int[]
    
    for i in 1:n_patients
        n_samples = labels[i] == 1 ? samples_per_positive : samples_per_negative
        patient_features = extract_patient_features(
            data[i], n_timeframes[i], spectrum_length, n_samples
        )
        append!(fixed_samples, patient_features)
        append!(fixed_labels, fill(labels[i], n_samples))
    end
    
    return fixed_samples, fixed_labels
end

"""
    prepare_timeframe_features(data::Vector{Vector{Matrix{Float64}}}, 
                              n_timeframes::Vector{Int},
                              spectrum_length::Int)

Extract features for all timeframes from all patients.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: Vector of patient data (each patient has vector of timeframe matrices)
- `n_timeframes`: Vector of timeframe counts per patient
- `spectrum_length`: Length of the spectrum

# Returns
- Vector of extracted features from all timeframes
"""
function prepare_timeframe_features(data::Vector{Vector{Matrix{Float64}}}, 
                                   n_timeframes::Vector{Int},
                                   spectrum_length::Int)
    features = []
    for (patient_idx, patient_data) in enumerate(data)
        for timeframe_idx in 1:n_timeframes[patient_idx]
            feature = extract_spectrum_features(patient_data[timeframe_idx], spectrum_length)
            push!(features, feature)
        end
    end
    return features
end

end