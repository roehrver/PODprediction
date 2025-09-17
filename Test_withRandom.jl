"""
Example usage of the refactored Risk Evaluation system.
This script demonstrates on randomly generated data, how to use the classification components. It was mostly written by Claude Opus 4.1.
"""

include(homedir() * "/PhD/temPODPrediction/DataPrep.jl")
include(homedir() * "/PhD/temPODPrediction/FeatureExtraction.jl")
include(homedir() * "/PhD/temPODPrediction/Classification.jl")
include(homedir() * "/PhD/temPODPrediction/Utility.jl")

# Load the main module
using .DataPrep
using .FeatureExtraction
using .Classification
using .Utilities

using DataFrames
using LinearAlgebra
using PosDefManifold

#=
Example 1: Using the Balanced Random Forest Classifier
=#
function example_brfc_classification()
    println("\n" * "="^60)
    println("Example 1: Balanced Random Forest Classification")
    println("="^60)
    
    # Create sample data
    n_samples = 100
    n_features = 10
    
    # Create sample DataFrame with features
    train_df = DataFrame(rand(n_samples, n_features), :auto)
    
    # Add medication columns
    train_df.desflurane = rand([0, 1], n_samples)
    train_df.sevoflurane = rand([0, 1], n_samples) .* (1 .- train_df.desflurane)
    train_df.propofol = 1 .- train_df.desflurane .- train_df.sevoflurane

    # Create labels (with imbalance)
    train_classes = [i <= 30 ? 1 : 0 for i in 1:n_samples]
    
    # Create test data
    test_df = similar(train_df[1:20, :])
    for col in names(test_df)
        test_df[!, col] = rand(20)
    end
    
    # Create interaction features
    train_with_interactions = create_key_interactions(
        train_df, 
        ["desflurane", "sevoflurane", "propofol"]
    )
    test_with_interactions = create_key_interactions(
        test_df,
        ["desflurane", "sevoflurane", "propofol"]
    )
    
    println("Training data shape: ", size(train_with_interactions))
    println("Test data shape: ", size(test_with_interactions))
    
    # Run classification
    test_pred, train_pred, ext_pred = balanced_random_forest_prediction(
        train_df, 
        train_classes, 
        test_df,
        key_columns = ["desflurane", "sevoflurane", "propofol"]
    )
    
    println("Training predictions (first 5): ", train_pred[1:5])
    println("Test predictions (first 5): ", test_pred[1:5])
    
    # Calculate metrics
    metrics = calculate_performance_metrics(train_classes, train_pred)
    print_performance_summary(metrics; model_name="BRFC Training")
end

example_brfc_classification()

#=
Example 2: Covariance-based SVM Classification
=#
function example_covariance_classification()
    println("\n" * "="^60)
    println("Example 2: Covariance-based SVM Classification")
    println("="^60)
    
    # Create synthetic covariance data
    n_patients_train = 50
    n_patients_test = 10
    n_channels = 8
    min_timeframes = 10
    max_timeframes = 30
    
    # Generate training data
    train_data = Vector{Vector{Hermitian}}()
    train_classes = Int[]
    num_timeframes_train = Int[]
    mean_per_patient = Hermitian[]
    
    for i in 1:n_patients_train
        n_timeframes = rand(min_timeframes:max_timeframes)
        patient_covs = Hermitian[]
        
        # Generate covariances for this patient
        for j in 1:n_timeframes
            # Create positive definite matrix
            A = randn(n_channels, n_channels)
            cov = Hermitian(A * A' + I)
            push!(patient_covs, cov)
        end
        
        push!(train_data, patient_covs)
        push!(train_classes, i <= 15 ? 1 : 0)  # Imbalanced classes
        push!(num_timeframes_train, n_timeframes)
    end
    
    # Generate test data
    test_data = Vector{Vector{Hermitian}}()
    test_classes = Int[]
    num_timeframes_test = Int[]
    mean_per_test = Hermitian[]
    
    for i in 1:n_patients_test
        n_timeframes = rand(min_timeframes:max_timeframes)
        patient_covs = Hermitian[]
        
        for j in 1:n_timeframes
            A = randn(n_channels, n_channels)
            cov = Hermitian(A * A' + I)
            push!(patient_covs, cov)
        end
        
        push!(test_data, patient_covs)
        push!(test_classes, i <= 3 ? 1 : 0)
        push!(num_timeframes_test, n_timeframes)
    end
    
    # Create external validation data
    external_data = test_data[1:5]  # Use subset as external
    
    println("Training: $(n_patients_train) patients, $(sum(train_classes)) POD cases")
    println("Test: $(n_patients_test) patients, $(sum(test_classes)) POD cases")
    
    # Run classification (reduced parameters for example)
    test_pred, train_pred, external_pred = Classification.covariance_prediction(
        train_data,
        test_data,
        train_classes,
        external_data,
        nothing;  # external_mean
        num_timeframes_test = num_timeframes_test,
        num_timeframes_train = num_timeframes_train,
        n_bags = 2,  # Reduced for example
        n_samples_majority = 5,
        undersampling_params = [0.3, 2],
    )
    
    println("\nPredictions:")
    println("Train (first 5): ", train_pred[1:min(5, length(train_pred))])
    println("Test (first 5): ", test_pred[1:min(5, length(test_pred))])
    
    # Calculate and display metrics
    train_metrics = Utilities.calculate_performance_metrics(train_classes, train_pred)
    test_metrics = Utilities.calculate_performance_metrics(test_classes, test_pred)

    Utilities.print_performance_summary(train_metrics; model_name="Covariance SVM Training")
    Utilities.print_performance_summary(test_metrics; model_name="Covariance SVM Test")
end

example_covariance_classification()

#=
Example 3: Spectrum-based Classification
=#
function example_spectrum_classification()
    println("\n" * "="^60)
    println("Example 3: Spectrum-based SVM Classification")
    println("="^60)
    
    # Create synthetic spectrum data
    n_patients_train = 40
    n_patients_test = 10
    spectrum_length = 46
    n_channels = 10
    min_timeframes = 10
    max_timeframes = 25
    
    # Generate training data
    train_data = Vector{Vector{Matrix{Float64}}}()
    train_classes = Int[]
    num_timeframes_train = Int[]
    
    for i in 1:n_patients_train
        n_timeframes = rand(min_timeframes:max_timeframes)
        patient_spectra = Matrix{Float64}[]
        
        for j in 1:n_timeframes
            # Create spectrum matrix (spectrum_length x n_channels)
            spectrum = rand(spectrum_length, n_channels)
            push!(patient_spectra, spectrum)
        end
        
        push!(train_data, patient_spectra)
        push!(train_classes, i <= 12 ? 1 : 0)
        push!(num_timeframes_train, n_timeframes)
    end
    
    # Generate test data
    test_data = Vector{Vector{Matrix{Float64}}}()
    test_classes = Int[]
    num_timeframes_test = Int[]
    
    for i in 1:n_patients_test
        n_timeframes = rand(min_timeframes:max_timeframes)
        patient_spectra = Matrix{Float64}[]
        
        for j in 1:n_timeframes
            spectrum = rand(spectrum_length, n_channels)
            push!(patient_spectra, spectrum)
        end
        
        push!(test_data, patient_spectra)
        push!(test_classes, i <= 3 ? 1 : 0)
        push!(num_timeframes_test, n_timeframes)
    end
    
    # External validation
    external_data = test_data[1:5]
    external_mean = mean(external_data[1][1])
    println("External mean size: ", size(external_mean))
    
    println("Training: $(n_patients_train) patients, $(sum(train_classes)) POD cases")
    println("Test: $(n_patients_test) patients, $(sum(test_classes)) POD cases")
    
    # Define dummy medication data
    medication_train = ones(Int, n_patients_train)
    medication = ones(Int, n_patients_test)

    
    # Run spectrum classification with corrected keyword arguments
   # Run spectrum classification with correct positional arguments
    test_pred, train_pred, external_pred = spectrum_prediction(
        train_data,
        test_data,
        train_classes,
        external_data,  # test_epod
        external_mean,  # baseline_mean
        medication,
        medication_train,
        num_timeframes_test,
        num_timeframes_train;
        n_bags = 2,  # Reduced for example
        n_samples = 5,
        spectrum_length = spectrum_length,
        undersampling_params = [0.3, 2],
        fold = 1,
        save_results = false
    )
    
    println("\nPredictions:")
    println("Train (first 5): ", train_pred[1:min(5, length(train_pred))])
    println("Test (first 5): ", test_pred[1:min(5, length(test_pred))])
    
    # Calculate metrics
    train_metrics = calculate_performance_metrics(train_classes, train_pred)
    test_metrics = calculate_performance_metrics(test_classes, test_pred)
    
    print_performance_summary(train_metrics; model_name="Spectrum SVM Training")
    print_performance_summary(test_metrics; model_name="Spectrum SVM Test")
end

example_spectrum_classification()