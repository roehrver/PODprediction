module PreProcessing

using Statistics, LinearAlgebra, DSP, FourierAnalysis, PosDefManifold, CovarianceEstimation

# Import from refactored modules
using Main.Segment, Main.ArtifactFiltering

"""
PreProcessing Module
Provides burst suppression estimation and covariance/spectrum timeline analysis for EEG data.
Refactored by Claude Opus 4.1 for improved readability and performance.
"""

export burst_suppression_estimation,
    burst_suppression_estimation_all_channels,
    covariance_timeline,
    covariance_timeline_no_bs,
    covariance_timeline_3channels,
    spectrum_timeline,
    segment_smooth

"""
    segment_smooth(signal::Vector{<:Float64}, is_valid::Vector{<:Float64}, window_size::Int)

Apply moving average smoothing within valid segments only.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `signal`: Input signal to smooth
- `is_valid`: Binary vector indicating valid samples (1) vs artifacts (0)
- `window_size`: Half-width of smoothing window in samples

# Returns
- Smoothed signal with same length as input
"""
function segment_smooth(signal::Vector{<:Float64}, is_valid::Vector{<:Int}, window_size::Int)
    smoothed = zeros(eltype(signal), length(signal))
    
    # Find segment boundaries
    breaks = [1; findall(diff(is_valid) .> 0) .+ 1; length(is_valid) + 1]

    # Process each valid segment
    for seg_idx in 1:(length(breaks)-1)
        seg_start = breaks[seg_idx]
        seg_end = breaks[seg_idx+1] - 1

        # Only process valid segments
        if seg_start <= length(is_valid) && is_valid[seg_start] == 1
            seg_length = seg_end - seg_start + 1
            
            if seg_length <= 2 * window_size
                # Small segments: use simple mean
                seg_mean = mean(@view signal[seg_start:seg_end])
                smoothed[seg_start:seg_end] .= seg_mean
            else
                # Large segments: efficient moving average using cumsum
                seg_data = @view signal[seg_start:seg_end]
                cumsum_data = cumsum(seg_data)
                
                for i in 1:seg_length
                    local_start = max(1, i - window_size)
                    local_end = min(seg_length, i + window_size)
                    
                    if local_start == 1
                        smoothed[seg_start + i - 1] = cumsum_data[local_end] / (local_end - local_start + 1)
                    else
                        smoothed[seg_start + i - 1] = (cumsum_data[local_end] - cumsum_data[local_start - 1]) / 
                                                      (local_end - local_start + 1)
                    end
                end
            end
        end
    end
    return smoothed
end

"""
    burst_suppression_estimation(data::Matrix, times::Any, sr::Float64, remove_indices::Vector{Int};
                                quantile_threshold::Float64=0.6, bs_threshold::Float64=0.8, channel::Int=3)

Estimate burst suppression features for a single channel.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: EEG data matrix (samples × channels)
- `times`: Time vector
- `sr`: Sampling rate in Hz
- `remove_indices`: Indices marked as artifacts
- `quantile_threshold`: Quantile for amplitude thresholding (set: 0.5-0.6)
- `bs_threshold`: Threshold for burst suppression detection (default: 0.8)
- `channel`: Channel to analyze (default: 3)

# Returns
- `bs_timeline`: Burst suppression probability timeline
- `bs_ratio`: Burst suppression ratio
- `longest_bs_period`: Longest burst suppression period in seconds
- `total_bs_time`: Total burst suppression time in seconds
"""
function burst_suppression_estimation(data::Matrix, times::Any, sr::Float64, 
                                     remove_indices::Vector{Int};
                                     quantile_threshold::Float64=0.6, 
                                     bs_threshold::Float64=0.8, 
                                     channel::Int=3)
    
    # Get valid (non-artifact) indices
    valid_indices = setdiff(1:length(times), remove_indices)
    
    # Find low amplitude samples
    channel_data = @view data[:, channel]
    amplitude_threshold = quantile(abs.(channel_data[valid_indices]), quantile_threshold)
    low_amplitude_indices = findall(abs.(channel_data) .< amplitude_threshold)
    
    # Create binary burst suppression indicator
    bs_indicator = zeros(length(channel_data))
    bs_indicator[low_amplitude_indices] .= 1.0
    
    # Create validity mask
    is_valid = Int.(zeros(length(times)))
    is_valid[valid_indices] .= 1
    
    # Smooth within valid segments (±0.5 second window)
    window_size = Int(round(sr * 0.5))
    bs_timeline = segment_smooth(bs_indicator, is_valid, window_size)
    
    # Calculate metrics on valid samples only
    valid_bs = bs_timeline[valid_indices]
    bs_indices = findall(valid_bs .> bs_threshold)
    
    # Compute statistics
    bs_ratio = length(bs_indices) / length(valid_indices)
    total_bs_time = length(bs_indices) / sr
    
    # Find longest burst suppression period
    longest_bs_period = find_longest_bs_period(valid_bs .> bs_threshold) / sr
    
    return bs_timeline, bs_ratio, longest_bs_period, total_bs_time
end

"""
    burst_suppression_estimation_all_channels(data::Matrix, times::Any, sr::Float64, 
                                             remove_indices::Vector{Int}, amplitude_mean::Float64;
                                             quantile_threshold::Float64)

Estimate burst suppression across all channels, requiring BS in all channels simultaneously.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: EEG data matrix (samples × channels)
- `times`: Time vector
- `sr`: Sampling rate in Hz
- `remove_indices`: Indices marked as artifacts
- `amplitude_mean`: Mean amplitude for threshold calculation
- `quantile_threshold`: Quantile for thresholding (set: 0.5-0.6) (0 uses mean amplitude)

# Returns
- `bs_timeline`: Average burst suppression timeline across channels
- `bs_ratio`: Burst suppression ratio
- `longest_bs_period`: Longest burst suppression period
- `total_bs_time`: Total burst suppression time
"""
function burst_suppression_estimation_all_channels(data::Matrix, times::Any, sr::Float64,
                                                  remove_indices::Vector{Int}, amplitude_mean::Float64,
                                                  quantile_threshold::Float64)
    
    n_samples, n_channels = size(data)
    valid_indices = setdiff(1:length(times), remove_indices)
    
    # Create validity mask
    is_valid = Int.(zeros(length(times)))
    is_valid[valid_indices] .= 1
    
    # Process each channel
    channel_bs_timelines = Vector{Vector{Float64}}()
    bs_threshold = 0.99 - 0.99 / amplitude_mean
    window_size = Int(round(sr * 0.5))
    
    # Track indices where ALL channels show burst suppression
    all_channels_bs = trues(n_samples)
    all_channels_bs_valid = trues(length(valid_indices))
    
    for ch in 1:n_channels
        channel_data = @view data[:, ch]
        
        # Determine threshold
        if quantile_threshold > 0
            threshold = quantile(abs.(channel_data[valid_indices]), quantile_threshold)
        else
            threshold = mean(abs.(channel_data[valid_indices]))
        end
        
        println("Channel $ch amplitude threshold: $threshold")
        
        # Find low amplitude samples
        low_amplitude = abs.(channel_data) .< threshold
        
        # Create and smooth BS indicator
        bs_indicator = zeros(n_samples)
        bs_indicator[low_amplitude] .= 1.0
        bs_timeline = segment_smooth(bs_indicator, is_valid, window_size)
        
        push!(channel_bs_timelines, bs_timeline)
        
        # Update all-channel BS tracking
        channel_bs = bs_timeline .> bs_threshold
        all_channels_bs .&= channel_bs
        all_channels_bs_valid .&= channel_bs[valid_indices]
    end
    
    # Average BS timeline across channels
    bs_timeline_avg = reduce(.+, channel_bs_timelines) ./ n_channels
    
    # Calculate metrics based on simultaneous BS in all channels
    bs_count = sum(all_channels_bs_valid)
    total_bs_time = bs_count / sr
    bs_ratio = bs_count / length(valid_indices)
    longest_bs_period = find_longest_bs_period(all_channels_bs_valid) / sr
    
    return bs_timeline_avg, bs_ratio, longest_bs_period, total_bs_time
end

"""
    find_longest_bs_period(bs_binary::BitVector)

Find the longest consecutive burst suppression period.
Refactored by Claude Opus 4.1 for improved readability and performance.
"""
function find_longest_bs_period(bs_binary::BitVector)
    if isempty(bs_binary) || !any(bs_binary)
        return 0
    end
    
    max_length = 0
    current_length = 0
    
    for is_bs in bs_binary
        if is_bs
            current_length += 1
            max_length = max(max_length, current_length)
        else
            current_length = 0
        end
    end
    
    return max_length
end

"""
    covariance_timeline(data::Matrix, times::Any, sr::Float64, bs_timeline::Vector,
                       remove_indices::Vector{Int}; 
                       intervals::Vector=[2, 4, 8, 12, 15, 20, 30, 40],
                       segment_seconds::Float64=5)

Compute covariance matrices over time with burst suppression as additional feature.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: EEG data matrix
- `times`: Time vector
- `sr`: Sampling rate
- `bs_timeline`: Burst suppression timeline
- `remove_indices`: Artifact indices
- `intervals`: Frequency band edges for filtering
- `segment_seconds`: Segment length in seconds

# Returns
- `covariances`: Vector of Hermitian covariance matrices
- `segmented_data`: Segmented data arrays
"""
function covariance_timeline(data::Matrix, times::Any, sr::Float64, bs_timeline::Vector,
                            remove_indices::Vector{Int};
                            intervals::Vector=[2, 4, 8, 12, 15, 20, 30, 46],
                            segment_seconds::Real=5)
    
    # Apply bandpass filtering with 2-channel rotation
    filtered_data = bandpass_segment_2channels(data, intervals, sr)
    
    # Add burst suppression as additional feature
    n_samples = size(filtered_data, 1)
    data_with_bs = hcat(filtered_data, bs_timeline[1:n_samples])
    
    # Segment the data
    segmented_data, segmented_times = filter_segments_to_array(
        data_with_bs, times, remove_indices, sr; 
        max_segment_seconds=segment_seconds
    )
    
    # Compute covariance matrices with shrinkage
    shrinkage_method = LinearShrinkage(target=DiagonalCommonVariance(), shrinkage=:oas)
    covariances = ℍVector([
        Hermitian(cov(shrinkage_method, segment)) 
        for segment in segmented_data
    ])
    
    return covariances
end

"""
    covariance_timeline_no_bs(data::Matrix, times::Any, sr::Float64, 
                             remove_indices::Vector{Int};
                             intervals::Vector=[2, 4, 8, 12, 15, 20, 30, 40],
                             segment_seconds::Float64=2)

Compute covariance matrices without burst suppression feature.
Refactored by Claude Opus 4.1 for improved readability and performance.
"""
function covariance_timeline_no_bs(data::Matrix, times::Any, sr::Float64,
                                  remove_indices::Vector{Int};
                                  intervals::Vector=[2, 4, 8, 12, 15, 20, 30, 46],
                                  segment_seconds::Real=2)
    
    # Apply bandpass filtering
    filtered_data = bandpass_segment_2channels(data, intervals, sr)
    
    # Segment the data
    segmented_data, segmented_times = filter_segments_to_array(
        filtered_data, times, remove_indices, sr;
        max_segment_seconds=segment_seconds
    )
    
    # Compute covariance matrices
    shrinkage_method = LinearShrinkage(target=DiagonalCommonVariance(), shrinkage=:oas)
    covariances = ℍVector([
        Hermitian(cov(shrinkage_method, segment))
        for segment in segmented_data
    ])

    return covariances
end

"""
    covariance_timeline_3channels(data::Matrix, times::Any, sr::Float64, bs_timeline::Vector,
                                 remove_indices::Vector{Int};
                                 intervals::Vector=[2, 4, 8, 12, 15, 20, 30, 40],
                                 segment_seconds::Float64=5)

Compute covariance matrices using 3-channel rotation.
Refactored by Claude Opus 4.1 for improved readability and performance.
"""
function covariance_timeline_3channels(data::Matrix, times::Any, sr::Float64, bs_timeline::Vector,
                                      remove_indices::Vector{Int};
                                      intervals::Vector=[2, 4, 8, 12, 15, 20, 30, 46],
                                      segment_seconds::Real=5)
    
    # Apply bandpass filtering with 3-channel rotation
    filtered_data = bandpass_segment_3channels(data, intervals, sr)
    
    # Segment the data
    segmented_data, segmented_times = filter_segments_to_array(
        filtered_data, times, remove_indices, sr;
        max_segment_seconds=segment_seconds
    )
    
    # Compute covariance matrices
    shrinkage_method = LinearShrinkage(target=DiagonalCommonVariance(), shrinkage=:oas)
    covariances = ℍVector([
        Hermitian(cov(shrinkage_method, segment))
        for segment in segmented_data
    ])
    
    return covariances, segmented_data
end

"""
    spectrum_timeline(data::Matrix, times::Any, sr::Float64,
                     remove_indices::Vector{Int};
                     segment_seconds::Float64=5)

Compute power spectrum timeline from segmented data.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: EEG data matrix
- `times`: Time vector  
- `sr`: Sampling rate in Hz
- `remove_indices`: Artifact indices to remove
- `segment_seconds`: Segment length in seconds (default: 5)

# Returns
- Vector of power spectra for each segment
"""
function spectrum_timeline(data::Matrix, times::Any, sr::Float64,
                          remove_indices::Vector{Int};
                          segment_seconds::Real=5)
    
    # Segment the data
    segmented_data, segmented_times = filter_segments_to_array(
        data, times, remove_indices, sr;
        max_segment_seconds=segment_seconds
    )
    
    # Compute power spectra for each segment
    sr_int = Int(round(sr))
    spectra_list = [
        spectra(segment, sr_int, sr_int; func=√).y 
        for segment in segmented_data
    ]
    
    return spectra_list
end

end