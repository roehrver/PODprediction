# Refactored by Claude Opus 4.1
# EEG data processing pipeline with burst suppression analysis

push!(LOAD_PATH, homedir() * "/PhD/PODprediction")
include(homedir() * "/PhD/Segment.jl")
include(homedir() * "/PhD/PreProcessing.jl")
include(homedir() * "/PhD/ArtifactFiltering.jl")

using Main.Segment, Main.PreProcessing, Main.ArtifactFiltering
using DataFrames
using CSV
using Statistics
using LinearAlgebra
using DSP
using PyCall

# Global filter settings
const HPF = 0.3  # High-pass filter cutoff (Hz)
const LPF = 48   # Low-pass filter cutoff (Hz)

# Data paths
eeg_data_path = homedir() * "/path/to/EEGdata"
clinical_data_path = homedir() * "/path/to/clinicaldata"

# Load clinical data
clinical_df = CSV.read(clinical_data_path, DataFrame)

# Initialize new columns for analysis results
clinical_df.bsr = missings(Float64, nrow(clinical_df))
clinical_df.longest_suppression = missings(Float64, nrow(clinical_df))
clinical_df.burst_suppression_ratio = missings(Float64, nrow(clinical_df))
clinical_df.recording_length = missings(Float64, nrow(clinical_df))
clinical_df.artifact_free_duration = missings(Float64, nrow(clinical_df))
clinical_df.mean_amplitude = missings(Float64, nrow(clinical_df))
clinical_df.sampling_rate = missings(Float64, nrow(clinical_df))
clinical_df.error_msg = missings(String, nrow(clinical_df))

# Initialize storage for processed data
n_cases = nrow(clinical_df)
covariance_matrices = Vector{Vector{Hermitian{Float64}}}(undef, n_cases)
power_spectra = Vector{Vector{Matrix{Float64}}}(undef, n_cases)
covariance_no_burst_suppression = Vector{Vector{Hermitian{Float64}}}(undef, n_cases)

# Track failed processing attempts
failed_indices = Int[]

# Get all valid folder paths
folder_paths = filter(x -> isdir(joinpath(eeg_data_path, x)), readdir(eeg_data_path))

# Process each patient folder
for (idx, folder_path) in enumerate(folder_paths)
    patient_id = folder_path[1:7]
    
    println("Processing patient: $patient_id (index: $idx)")
    
    # TODO: Load EEG data from folder
    # This should load: data (raw EEG matrix), times (time vector), sr (sampling rate)
    # Example: data, times, sr = load_eeg_data(joinpath(eeg_data_path, folder_path))
    
    try
        # Filter artifacts from raw data
        filtered_data, times, artifacts_to_remove = ArtifactFiltering.filter_artifacts(
            data, 
            times, 
            sr, 
            threshold=70
        )
        
        n_samples, n_channels = size(filtered_data)
        clean_indices = setdiff(1:length(times), artifacts_to_remove)
        
        # Apply bandpass filter
        bandpassed_data = ArtifactFiltering.bandpass_segment_to_channels(
            filtered_data, 
            [HPF, LPF], 
            sr
        )
        
        # Calculate mean amplitude for normalization
        mean_amplitude = mean(abs.(bandpassed_data[clean_indices, :]))
        mean_amplitude = max(1.5, mean_amplitude)
        
        # Estimate burst suppression
        burst_segments = []
        try
            # Normalize data for burst suppression detection
            normalized_data = bandpassed_data * max(1, (1.5 / mean_amplitude))
            
            burst_segments, bsr, longest_suppression_period, burst_suppression_duration = 
                Main.PreProcessing.burst_suppression_estimation_all_channels(
                    normalized_data, 
                    times, 
                    sr, 
                    artifacts_to_remove, 
                    mean_amplitude, 
                    0.55
                )
            
            # Store results in dataframe
            clinical_df.bsr[idx] = bsr
            clinical_df.longest_suppression[idx] = longest_suppression_period
            clinical_df.burst_suppression_ratio[idx] = burst_suppression_duration
            clinical_df.artifact_free_duration[idx] = length(clean_indices) / sr
            clinical_df.mean_amplitude[idx] = mean_amplitude
            clinical_df.sampling_rate[idx] = sr
            
        catch e
            clinical_df.error_msg[idx] = "Burst suppression estimation failed: $(e)"
            push!(failed_indices, idx)
            continue
        end
        
        # Re-reference data using common average
        rereferenced_data = ArtifactFiltering.rereference_by_common_average(filtered_data)
        rereferenced_bandpassed = ArtifactFiltering.bandpass_segment_to_channels(
            rereferenced_data, 
            [HPF, LPF], 
            sr
        )
        
        # Calculate covariance matrices across time
        try
            frequency_bands = [HPF, 4, 8, 12, 15, 20, 30, LPF]
            covariance_timeline = PreProcessing.covariance_timeline(
                rereferenced_data, 
                times, 
                sr, 
                burst_segments, 
                artifacts_to_remove,
                intervals=frequency_bands,
                segment_seconds=30
            )
            covariance_matrices[idx] = reverse(covariance_timeline)
            println("Covariance matrix size: $(size(covariance_timeline[1]))")
            
        catch e
            clinical_df.error_msg[idx] = "Covariance calculation failed: $(e)"
            push!(failed_indices, idx)
            continue
        end
        
        # Calculate power spectrum timeline
        try
            spectrum_timeline = PreProcessing.spectrum_timeline(
                rereferenced_bandpassed, 
                times, 
                sr, 
                artifacts_to_remove, 
                segment_seconds=30
            )
            power_spectra[idx] = reverse(spectrum_timeline)
            
        catch e
            clinical_df.error_msg[idx] = "Spectrum calculation failed: $(e)"
            push!(failed_indices, idx)
            continue
        end
        
    catch e
        clinical_df.error_msg[idx] = "General processing error: $(e)"
        push!(failed_indices, idx)
    end
end

# Remove failed entries
unique!(failed_indices)
sort!(failed_indices, rev=true)

println("Failed processing for $(length(failed_indices)) cases")

# TODO: save the results, excluding undefined entries
