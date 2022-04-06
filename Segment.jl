module Segment

using StatsBase,
    Statistics,
    LinearAlgebra,
    DSP,
    FourierAnalysis,
    YAML,
    PosDefManifold,
    PosDefManifoldML,
    PyCall,
    MultivariateStats

"""
Segment module:
Filters out some artifacts, calculates bandpass filter &
segments data
"""

export artifactfilter,
    filterSegmentsToarray,
    bandpassSegmentToChannels,
    bandpassSegmentChannels,
    bandpassSegment2Channels


"""
artifactfilter(data, times, sr, quant)

Set high amplitude artifacts to 0 and add buffer around the time points.

returns data, times, remove_index (to be removed after bandpass filtering)

"""
function artifactfilter(data, times, sr, quant)
    ns, ne = size(data)
    cutoff = []
    ampl = Int[]
    l = length(data[:, 1])
    data_mean = [mean(data[i, :]) for i = 1:length(times)]
    #filter out when recording continued after OP stopped, in next iteration cur to length of OP
    data_dif = [
        1 / mean(data_mean[i+1, :] - data_mean[i, :]) for i = 1:length(times)-1
    ]
    q = max(quantile!(data_dif, 0.925), 10)
    append!(ampl, findall(data_dif .> q))
    #high amplitude filter
    ampl = sort(unique(ampl))
    ampl = ampl[findall(ampl .> length(times) * 0.85)]
    data[ampl, :] = zeros(length(ampl), ne)
    for i = 1:ne
        q = quantile!(data[:, i], quant)
        append!(cutoff, q)
    end
    for i = 1:ne
        ampl = append!(ampl, findall(data[:, i] .> cutoff[i]))
        ampl = append!(ampl, findall(data[:, i] .< -cutoff[i]))
    end
    remove_index = unique(ampl)
    #set to 0
    data[remove_index, :] = zeros(length(remove_index), ne)
    #add buffer
    l = length(remove_index)
    last = length(data[:, 1])
    for i = 1:l
        for j =
            Int(
                maximum([remove_index[i] - Int(floor(sr / 2)), 1]),
            ):Int(minimum([remove_index[i] + Int(floor(sr / 2)), last]))
            append!(remove_index, j)
        end
        if length(remove_index) > 3 * last
            unique!(remove_index)
        end
    end
    remove_index = unique(remove_index)
    remove_index = sort(remove_index)
    return data, times, remove_index
end

"""
filterSegmentsToarray(data, times, remove_afterbandpass, sr)

Automatically segment data by removing buffers to remove artifacts.
(amplitude artifacts and bandpass edge artifacts)
Segments smaller than 1 s are removed und segments longer than 1 minute are parted.

return segmenteddata, segmentedtimes
"""
function filterSegmentsToarray(data, times, remove_afterbandpass, sr)
    (ns, ne) = size(data)
    #println(size(data))
    keep = setdiff(1:length(times), remove_afterbandpass)
    data = data[keep, :]
    times = times[keep]
    #find breaks
    breaks = Int[1]
    prev = keep[1] - 1
    indexk = 1
    for k in keep
        if k - prev > 1
            append!(breaks, indexk)
        end
        prev = k
        indexk = indexk + 1
    end
    segmentedtimes = [times[breaks[i]:breaks[i+1]-1] for i = 1:length(breaks)-1]
    segmenteddata =
        [data[breaks[i]:breaks[i+1]-1, :] for i = 1:length(breaks)-1]
    tooshort = Int[]
    for i = 1:length(segmentedtimes)
        if length(segmentedtimes[i]) < sr
            append!(tooshort, i)
        end
        dt = fit(ZScoreTransform, segmenteddata[i], dims = 2)
        segmenteddata[i] = StatsBase.transform(dt, segmenteddata[i])
    end
    deleteat!(segmenteddata, tooshort)
    deleteat!(segmentedtimes, tooshort)
    for i = 1:length(segmentedtimes)
        if length(segmentedtimes[i]) > 60 * sr
            newl = Int(floor(length(segmentedtimes[i]) / 2))
            temp = segmentedtimes[i]
            insert!(
                segmentedtimes,
                i + 1,
                segmentedtimes[i][newl+1:length(segmentedtimes[i])],
            )
            segmentedtimes[i] = temp[1:newl]
            temp = segmenteddata[i]
            insert!(
                segmenteddata,
                i + 1,
                segmenteddata[i][newl+1:length(segmentedtimes[i]), :],
            )
            segmenteddata[i] = temp[1:newl, :]
            segmenteddata[i+1] = temp[newl+1:size(temp)[1], :]
        end
    end
    return segmenteddata, segmentedtimes
end

"""
bandpassSegmentToChannels(data, intervalls, sr)

Bandpass filter to frequency intervals and stack data.

return data
"""
function bandpassSegmentToChannels(data, intervalls, sr)
    (ns, ne) = size(data)
    compiled_data = Matrix{Float64}(undef, ns, ne * (length(intervalls) - 1))
    for i = 1:length(intervalls)-1
        BPfilter = digitalfilter(
            Bandpass(intervalls[i], intervalls[i+1]; fs = sr),
            Butterworth(2),
        )
        databand = filtfilt(BPfilter, data)
        j = i * ne - (ne - 1)
        compiled_data[:, j:j+(ne-1)] = databand
    end

    data = compiled_data
    println(size(data))
    return data
end

"""
bandpassSegmentToChannels(data, intervalls, sr)

Bandpass filter to frequency intervals and stack data using 2 out of 4 channels,
rotating choice of the 2.

return data
"""
function bandpassSegment2Channels(data, intervalls, sr)
    (ns, ne) = size(data)
    compiled_data = Matrix{Float64}(undef, ns, 2 * (length(intervalls) - 1))
    for i = 1:length(intervalls)-1
        BPfilter = digitalfilter(
            Bandpass(intervalls[i], intervalls[i+1]; fs = sr),
            Butterworth(2),
        )
        j = mod(i, 4)
        if j == 0
            j = 4
        end
        databand = filtfilt(BPfilter, data[:, [mod(i, 4) + 1, j]])
        compiled_data[:, 2*i-1:2*i] = databand
    end

    data = compiled_data
    println(size(data))
    return data
end
end
