module PreProcessing

using StatsBase,
    Statistics,
    LinearAlgebra,
    DSP,
    FourierAnalysis,
    PosDefManifold,
    PosDefManifoldML,
    DataFrames,
    CovarianceEstimation

push!(LOAD_PATH, homedir() * "/Promotion/BCI/")
using Segment
export filterandrereference,
    burstsuppressionestimation, covariancetimeline, spectrumtimeline


"""
filterandrereference(data, times, sr)

This is the first function to be called on the raw data. It filters out high amplitude artifacts and rereferences the data.
This was used on 4 frontal channels (sr=128 an sr=250, depending on the monitor).
Using it for bigger data might cause memory issues.

returns referferences data, times, remove_afterbandpass (containing all the indices that contain artifacts+buffer for the bandpass filter)
"""
function filterandrereference(data, times, sr)
    data, times, remove_afterbandpass =
        Segment.artifactfilter(data, times, sr, 0.99)
    rereference = mean(data[:, i] for i = 1:ne)
    data = data .- rereference
    return data, times, remove_afterbandpass
end


"""
burstsuppressionestimation(data, times, sr, remove_afterbandpass)

Estimation of burst suppression features. Either do the bandpassfiltering here or
before, then data needs to be bandpass filtered.

returns b (burst suppresion timeline),
bsr (burst suppression ratio)
lbsp (longest burst suppression phase)
burstsupp (total suppression time)
"""
function burstsuppressionestimation(data, times, sr, remove_afterbandpass)
    dataS = bandpassSegmentToChannels(data, [0.3, 60], sr)#optional =data
    b = []
    temp = findall(abs.(dataS[:, 3]) .< quantile(abs.(dataS[:, 3]), 0.6))
    b = zeros(length(data[:, 3]))
    b[temp] .= 1
    b = [mean(b[max(1, i - 64):min(length(b), i + 64)]) for i = 1:length(b)]
    keep = setdiff(1:length(times), remove_afterbandpass)
    lbsp = zeros(length(b[keep]))
    lbsp[findall(b[keep] .> 0.8)] .= 1
    l = 1
    temp = 1
    for i = 1:length(lbsp)
        if lbsp[i] == 1# && i<length(lbsr) do not count BS, if it is til the last recorded timepoint that is unlikely
            temp = temp + 1
        else
            if temp > l
                l = temp
                println(l)
            end
            temp = 1
        end
    end
    bsr = length(findall(b[keep] .> 0.8)) / length(b[keep])
    lbsp = l / sr
    burstsupp = length(findall(b[keep] .> 0.8)) / sr
    return b, bsr, lbsr, burstsupp
end

"""
riemanianpotatovariant(C, segmentedtimes)

Outlier detection using riemanian geometry and the riemanian potato.
Ref: Barachant A, Andreev A, Congedo M. The riemannian potato: an automatic and adaptive artifact detection
method for online experiments using riemannian geometry. TOBI Workshop lV (2013).

returns C (covariance timeline with outliers removed),
segmetedtimes (the corresponding time frames),
ppp (outlier indices, for removal of timeframes for spectrum)

"""
function riemanianpotatovariant(C, segmentedtimes)
    potatomean = getMean(Fisher, C)
    potatodist =
        getDistances(Fisher, ℍVector([Hermitian(potatomean)]), ℍVector(C))
    potatodist = reshape(potatodist, size(potatodist)[2])
    ppp = findall(potatodist .> quantile!(potatodist, 0.95))
    thr = quantile!(potatodist, 0.975)
    count = 1
    potatomean = getMean(Fisher, C[1:min(length(potatodist), 180)])
    for i = 1:length(potatodist)
        d = distanceSqr(Fisher, potatomean, C[i])
        potatodist[i] = d
        if d < thr && !(i in ppp)
            potatomean = geodesic(Fisher, potatomean, C[i], 0.01)
            thr = quantile!(
                potatodist[max(1, i - 120):min(length(potatodist), i + 60)],
                0.975,
            )
        elseif !(i in ppp)
            count = count + 1
            append!(ppp, i)
        end
    end
    ppp = sort!(ppp)
    unique!(ppp)
    deleteat!(segmentedtimes, ppp)
    deleteat!(C, ppp)
    return C, segmentedtimes, ppp
end

"""
clustertimeframes(segmentedtimes, minutes)

Clusters timeframe indices by minute windows.

returns clustered indices
"""
function clustertimeframes(segmentedtimes, minutes)
    clusterbytimeframes = Int[1]
    timeframe = minutes * 60
    numberoftimeframe = ceil(segmentedtimes[1][1] / timeframe)
    for i = 1:length(segmentedtimes)
        if segmentedtimes[i][length(segmentedtimes[i])] >
           numberoftimeframe * timeframe && i > 1
            append!(clusterbytimeframes, i)
            numberoftimeframe = numberoftimeframe + 1
        end
    end
    if clusterbytimeframes[length(clusterbytimeframes)] < length(segmentedtimes)
        append!(clusterbytimeframes, length(segmentedtimes))
    end
    return clusterbytimeframes
end


"""
covariancetimeline(
    data,
    times,
    sr,
    b,
    remove_afterbandpass;
    minutes = 2,
)

Estimate covariances for timeframes of length minutes of bandpass-filtered
and stacked data. b is burst suppression timeline.

returns S (covariance timeline)
ppp (outlier indices, for removal of timeframes for spectrum)

"""
function covariancetimeline(
    data,
    times,
    sr,
    b,
    remove_afterbandpass;
    minutes = 2,
)
    data = bandpassSegment2Channels(data, [0.3, 4, 8, 12, 15, 20, 30, 60], sr)
    #add burst burst suppression timeline to stacked data
    dataB = zeros(size(data)[1], size(data)[2] + 1)
    dataB[:, 1:size(data)[2]] = data
    dataB[:, size(data)[2]+1] = b
    data = dataB

    segmenteddata, segmentedtimes =
        filterSegmentsToarray(data, times, remove_afterbandpass, sr)
    method =
        LinearShrinkage(target = DiagonalCommonVariance(), shrinkage = :oas)
    C = ℍVector([Hermitian(cov(method, X)) for X in segmenteddata])
    C, segmentedtimes, ppp = riemanianpotatovariant(C, segmentedtimes)
    clusterbytimeframes = clustertimeframes(segmentedtimes, minutes)

    S = Vector{Hermitian}(undef, size(clusterbytimeframes)[1] - 1)
    for j = 1:size(clusterbytimeframes)[1]-1
        l = minimum([clusterbytimeframes[j+1] - 1, size(C)[1]])
        tmp2 = ℍVector([C[i] for i = clusterbytimeframes[j]:l])
        temp = mean(Fisher, tmp2; verbose = false)
        S[j] = Hermitian(temp)
    end
    return S, ppp
end

"""
spectrumtimeline(
    data,
    times,
    sr,
    remove_afterbandpass;
    minutes = 2,
    ppp = [],
)

Estimate spectrum for timeframes of length minutes of bandpass-filtered data.
Removing the outlier segments calculated for the covariances (ppp)

returns S (spectrum timeline)

"""
function spectrumtimeline(
    data,
    times,
    sr,
    remove_afterbandpass;
    minutes = 2,
    ppp = [],
)
    dataS = bandpassSegmentToChannels(data, [0.3, 60], sr)
    dt = fit(ZScoreTransform, dataS, dims = 2)
    dataS = StatsBase.transform(dt, dataS)
    segmenteddata, segmentedtimes =
        filterSegmentsToarray(dataS, times, remove_afterbandpass, sr)
    deleteat!(segmentedtimes, ppp)
    deleteat!(segmenteddata, ppp)
    clusterbytimeframes = clustertimeframes(segmentedtimes, minutes)
    S = Vector{Matrix{Float64}}(undef, size(clusterbytimeframes)[1] - 1)
    for j = 1:size(clusterbytimeframes)[1]-1
        Σ = mean(
            spectra(segmenteddata[c], Int(sr), Int(sr); func = √).y for
            c = clusterbytimeframes[j]:clusterbytimeframes[j+1]
        )
        S[j] = Σ
    end
    return S
end
end
