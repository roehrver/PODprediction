module Segment

"""
Segment Module
Segments filtered EEG data by removing artifacts and creating fixed-length segments.
Refactored by Claude Sonnet 4 for improved readability and performance.
"""

export filter_segments_to_array

"""
    filter_segments_to_array(data::Matrix, times::Vector, remove_indices::Vector{Int}, sr::Real; 
                           max_segment_seconds::Real=30)

Segment continuous data by removing artifact indices and creating segments of appropriate length.
Segments shorter than 1 second are removed; segments longer than `max_segment_seconds` are split.
Refactored by Claude Sonnet 4 for improved readability and performance.

# Arguments
- `data`: Input data matrix (samples × channels)
- `times`: Time vector corresponding to data samples
- `remove_indices`: Indices to remove from data (artifacts and buffer zones)
- `sr`: Sampling rate in Hz
- `max_segment_seconds`: Maximum segment length in seconds (default: 30)

# Returns
- `segments_data`: Vector of data segment matrices
- `segments_times`: Vector of corresponding time vectors

# Notes
- Segments < 1 second are automatically removed
- Segments > max_segment_seconds are recursively split in half
- Z-scoring can be applied to segments after this function
"""
function filter_segments_to_array(data::Matrix, times::Any, remove_indices::Vector{Int}, 
                                 sr::Real; max_segment_seconds::Real=30)
    
    # Get valid indices (those not marked for removal)
    all_indices = 1:length(times)
    valid_indices = setdiff(all_indices, remove_indices)
    
    # Extract clean data
    clean_data = data[valid_indices, :]
    clean_times = times[valid_indices]
    
    # Find segment boundaries (where there are gaps in valid_indices)
    segment_breaks = find_segment_breaks(valid_indices)
    
    # Create initial segments
    segments_data = Vector{Matrix{eltype(data)}}()
    segments_times = Vector{Vector{eltype(times)}}()
    
    for i in 1:(length(segment_breaks) - 1)
        start_idx = segment_breaks[i]
        end_idx = segment_breaks[i + 1] - 1
        
        segment_data = clean_data[start_idx:end_idx, :]
        segment_time = clean_times[start_idx:end_idx]
        
        # Only keep segments >= 1 second
        if length(segment_time) >= sr
            push!(segments_data, segment_data)
            push!(segments_times, segment_time)
        end
    end
    
    # Split segments that are too long
    split_long_segments!(segments_data, segments_times, sr, max_segment_seconds)
    
    return segments_data, segments_times
end

"""
    find_segment_breaks(valid_indices::Vector{Int})

Find break points in a sequence of indices where gaps exist.
Refactored by Claude Sonnet 4 for improved readability and performance.

# Arguments
- `valid_indices`: Sorted vector of valid indices

# Returns
- Vector of break points (including start and end)
"""
function find_segment_breaks(valid_indices::Vector{Int})
    if isempty(valid_indices)
        return Int[1, 1]
    end
    
    # Find where consecutive differences > 1 (indicating gaps)
    gaps = findall(diff(valid_indices) .> 1)
    
    # Create break points: start, after each gap, and end
    breaks = [1; gaps .+ 1; length(valid_indices) + 1]
    
    return breaks
end

"""
    split_long_segments!(segments_data::Vector, segments_times::Vector, 
                        sr::Real, max_seconds::Real)

Split segments longer than max_seconds into smaller segments (in-place).
Refactored by Claude Sonnet 4 for improved readability and performance.

# Arguments
- `segments_data`: Vector of data segments (modified in-place)
- `segments_times`: Vector of time segments (modified in-place)
- `sr`: Sampling rate in Hz
- `max_seconds`: Maximum allowed segment length in seconds
"""
function split_long_segments!(segments_data::Vector, segments_times::Vector, 
                             sr::Real, max_seconds::Real)
    
    max_samples = Int(round(max_seconds * sr))
    i = 1
    
    while i <= length(segments_times)
        current_length = length(segments_times[i])
        
        # Check if segment needs splitting
        if current_length > max_samples
            # Calculate split point (middle of segment)
            mid_point = current_length ÷ 2
            
            # Create second half of segment
            second_half_data = segments_data[i][mid_point + 1:end, :]
            second_half_times = segments_times[i][mid_point + 1:end]
            
            # Trim first half
            segments_data[i] = segments_data[i][1:mid_point, :]
            segments_times[i] = segments_times[i][1:mid_point]
            
            # Insert second half after current segment
            insert!(segments_data, i + 1, second_half_data)
            insert!(segments_times, i + 1, second_half_times)
            
            # Don't increment i - recheck the current segment
            # in case it still needs further splitting
        else
            i += 1
        end
    end
end

end