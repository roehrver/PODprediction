module ArtifactFiltering

using Statistics, DSP

"""
Artifact Filtering Module
Filters EEG artifacts, applies bandpass filtering, and segments data.
Refactored by Claude Opus 4.1 for improved readability and performance.
"""

export filter_artifacts,
    bandpass_segment_to_channels,
    bandpass_segment_2channels,
    bandpass_segment_3channels,
    rereference_by_common_average

"""
    find_constant_indices(signal::Vector, min_length::Int)

Find indices where signal remains constant for at least `min_length` samples.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `signal`: Input signal vector
- `min_length`: Minimum length of constant segments to detect

# Returns
- Vector of indices belonging to constant segments
"""
function find_constant_indices(data::Any, min_length::Int)
    # Find run lengths of constant values
    change_points = findall(diff(data) .!= 0)
    runs = diff([0; change_points; size(data, 1)])

    # Calculate start indices for each run
    start_indices = cumsum([1; runs[1:end-1]])
    
    # Select runs meeting minimum length requirement
    constant_segments = [
        start:(start + run - 1) 
        for (start, run) in zip(start_indices, runs) 
        if run >= min_length
    ]
    
    return isempty(constant_segments) ? Int[] : vcat(constant_segments...)
end

"""
    rereference_by_common_average(data::Matrix)

Re-reference EEG data using common average reference.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: Matrix of EEG data (samples × channels)

# Returns
- Re-referenced data matrix
"""
function rereference_by_common_average(data::Any)
    common_average = mean(data, dims=2)
    return data .- common_average
end

"""
    peak_to_peak_rejection(channel::Vector, sr::Float64, threshold::Float64)

Detect artifacts using peak-to-peak amplitude in sliding windows.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `channel`: Single channel EEG data
- `sr`: Sampling rate in Hz
- `threshold`: Peak-to-peak amplitude threshold

# Returns
- Vector of indices to reject
"""
function peak_to_peak_rejection(channel::Any, sr::Float64, threshold::Real)
    window_size = Int(floor(0.2 * sr))  # 200ms window
    n_samples = length(channel)
    remove_indices = Int[]
    
    # Pre-allocate for better performance
    sizehint!(remove_indices, n_samples ÷ 4)
    
    for start_idx in 1:window_size:n_samples
        end_idx = min(start_idx + window_size - 1, n_samples)
        window_view = @view channel[start_idx:end_idx]
        
        # Check peak-to-peak amplitude
        if (maximum(window_view) - minimum(window_view)) > threshold
            append!(remove_indices, start_idx:end_idx)
        end
    end
    
    return unique!(remove_indices)
end

"""
    add_buffer_efficient!(remove_indices::Vector{Int}, buffer_samples::Int, total_samples::Int)

Add buffer around artifact indices with memory-efficient implementation.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `remove_indices`: Vector of indices to add buffer around (modified in-place)
- `buffer_samples`: Number of buffer samples to add on each side
- `total_samples`: Total number of samples in the data
"""
function add_buffer_efficient!(remove_indices::Vector{Int}, buffer_samples::Int, total_samples::Int)
    if isempty(remove_indices)
        return remove_indices
    end
    
    # Sort indices first for efficient processing
    sort!(remove_indices)
    
    # Create ranges with buffer
    buffered_ranges = Vector{UnitRange{Int}}()
    sizehint!(buffered_ranges, length(remove_indices))
    
    for idx in remove_indices
        start_idx = max(1, idx - buffer_samples)
        end_idx = min(total_samples, idx + buffer_samples)
        push!(buffered_ranges, start_idx:end_idx)
    end
    
    # Merge overlapping ranges
    merged_ranges = Vector{UnitRange{Int}}()
    current_range = buffered_ranges[1]
    
    for next_range in @view buffered_ranges[2:end]
        if first(next_range) <= last(current_range) + 1
            # Ranges overlap or are adjacent, merge them
            current_range = first(current_range):max(last(current_range), last(next_range))
        else
            push!(merged_ranges, current_range)
            current_range = next_range
        end
    end
    push!(merged_ranges, current_range)
    
    # Convert back to indices
    empty!(remove_indices)
    for range in merged_ranges
        append!(remove_indices, range)
    end
    
    return remove_indices
end

"""
    filter_artifacts(data::Matrix, times::Vector, sr::Float64; 
                    method::Symbol=:quantile, threshold::Float64=70, quantile_val::Float64=0.999)

Filter artifacts from EEG data using amplitude thresholding, plateau detection, and peak-to-peak rejection.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: EEG data matrix (samples × channels)
- `times`: Time vector
- `sr`: Sampling rate in Hz
- `method`: Thresholding method (:quantile or :fixed)
- `threshold`: Fixed threshold value (used when method=:fixed)
- `quantile_val`: Quantile value for threshold (used when method=:quantile)

# Returns
- `data`: Artifact-zeroed data
- `times`: Original time vector (unchanged)
- `remove_indices`: Indices marked as artifacts (including buffer)
"""
function filter_artifacts(data::Matrix, times::Any, sr::Float64; 
                         method::Symbol=:fixed, threshold::Real=70, quantile_val::Float64=0.999)
    
    n_samples, n_channels = size(data)
    remove_indices = Int[]
    
    # Pre-allocate for efficiency
    sizehint!(remove_indices, n_samples ÷ 10)
    
    # Process each channel
    for ch in 1:n_channels
        channel_data = @view data[:, ch]
        
        # 1. Amplitude thresholding
        if method == :quantile
            ch_threshold = quantile(abs.(channel_data), quantile_val)
        else
            ch_threshold = threshold
        end
        
        amplitude_artifacts = findall(abs.(channel_data) .> ch_threshold)
        append!(remove_indices, amplitude_artifacts)
        
        # 2. Plateau detection (constant values for >200ms)
        plateau_artifacts = find_constant_indices(channel_data, Int(floor(sr / 5)))
        append!(remove_indices, plateau_artifacts)
        
        # 3. Peak-to-peak rejection
        pp_threshold = method == :fixed ? threshold * 1.25 : 100
        pp_artifacts = peak_to_peak_rejection(channel_data, sr, pp_threshold)
        append!(remove_indices, pp_artifacts)
    end
    
    # Remove duplicates and sort
    unique!(remove_indices)
    sort!(remove_indices)
    
    # Zero out artifacts
    data[remove_indices, :] .= 0.0
    
    # Add buffer around artifacts (±0.5 seconds)
    buffer_samples = Int(floor(sr / 2))
    add_buffer_efficient!(remove_indices, buffer_samples, n_samples)
    
    return data, times, remove_indices
end

"""
    bandpass_segment_to_channels(data::Matrix, intervals::Vector, sr::Float64)

Apply bandpass filtering to frequency intervals and stack all channels.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: EEG data matrix (samples × channels)
- `intervals`: Frequency band edges for filtering
- `sr`: Sampling rate in Hz

# Returns
- Filtered and stacked data matrix
"""
function bandpass_segment_to_channels(data::Matrix, intervals::Vector, sr::Float64)
    n_samples, n_channels = size(data)
    n_bands = length(intervals) - 1
    
    # Pre-allocate output matrix
    output_data = Matrix{Float64}(undef, n_samples, n_channels * n_bands)
    
    for band_idx in 1:n_bands
        # Create bandpass filter
        bp_filter = digitalfilter(
            Bandpass(intervals[band_idx], intervals[band_idx + 1]; fs=sr),
            Butterworth(2)
        )
        
        # Apply filter to all channels
        filtered_data = filtfilt(bp_filter, data)
        
        # Store in output matrix
        col_start = (band_idx - 1) * n_channels + 1
        col_end = band_idx * n_channels
        output_data[:, col_start:col_end] = filtered_data
    end
    
    println("Output size: ", size(output_data))
    return output_data
end

"""
    bandpass_segment_2channels(data::Matrix, intervals::Vector, sr::Float64)

Apply bandpass filtering using rotating pairs of channels from 4-channel input.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: EEG data matrix (samples × 4 channels)
- `intervals`: Frequency band edges for filtering
- `sr`: Sampling rate in Hz

# Returns
- Filtered data with 2 channels per band
"""
function bandpass_segment_2channels(data::Matrix, intervals::Vector, sr::Float64)
    n_samples = size(data, 1)
    n_bands = length(intervals) - 1
    
    # Pre-allocate output matrix
    output_data = Matrix{Float64}(undef, n_samples, 2 * n_bands)
    
    for band_idx in 1:n_bands
        # Create bandpass filter
        bp_filter = digitalfilter(
            Bandpass(intervals[band_idx], intervals[band_idx + 1]; fs=sr),
            Butterworth(6)
        )
        
        # Select rotating channel pairs
        ch1 = mod1(band_idx + 1, 4)
        ch2 = mod1(band_idx, 4)
        
        # Apply filter to selected channels
        filtered_data = filtfilt(bp_filter, @view data[:, [ch1, ch2]])
        
        # Store in output matrix
        output_data[:, 2*band_idx - 1:2*band_idx] = filtered_data
    end
    
    println("Output size: ", size(output_data))
    return output_data
end

"""
    bandpass_segment_3channels(data::Matrix, intervals::Vector, sr::Float64)

Apply bandpass filtering using rotating triplets of channels from 4-channel input.
Refactored by Claude Opus 4.1 for improved readability and performance.

# Arguments
- `data`: EEG data matrix (samples × 4 channels)
- `intervals`: Frequency band edges for filtering
- `sr`: Sampling rate in Hz

# Returns
- Filtered data with 3 channels per band
"""
function bandpass_segment_3channels(data::Matrix, intervals::Vector, sr::Float64)
    n_samples = size(data, 1)
    n_bands = length(intervals) - 1
    
    # Pre-allocate output matrix
    output_data = Matrix{Float64}(undef, n_samples, 3 * n_bands)
    
    for band_idx in 1:n_bands
        # Create bandpass filter
        bp_filter = digitalfilter(
            Bandpass(intervals[band_idx], intervals[band_idx + 1]; fs=sr),
            Butterworth(2)
        )
        
        # Select rotating channel triplets
        ch1 = mod1(band_idx - 1, 4)
        ch2 = mod1(band_idx, 4)
        ch3 = mod1(band_idx + 1, 4)
        
        # Apply filter to selected channels
        filtered_data = filtfilt(bp_filter, @view data[:, [ch1, ch2, ch3]])
        
        # Store in output matrix
        output_data[:, 3*band_idx - 2:3*band_idx] = filtered_data
    end
    
    println("Output size: ", size(output_data))
    return output_data
end

end