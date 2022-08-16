module RiskEvaluation

using StatsBase,
    Statistics,
    LinearAlgebra,
    DSP,
    FourierAnalysis,
    YAML,
    PosDefManifold,
    PosDefManifoldML,
    PyCall,
    PyPlot,
    Random,
    CovarianceEstimation

using ScikitLearn
"""
Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: Machine
learning in Python. Journal of Machine Learning Research 12 (2011) 2825–2830.
"""
@sk_import metrics:balanced_accuracy_score
@sk_import metrics:roc_auc_score
@sk_import metrics:roc_curve
@sk_import svm:SVC
@sk_import preprocessing:StandardScaler
@sk_import model_selection:GridSearchCV
@sk_import model_selection:StratifiedKFold
@sk_import pipeline:make_pipeline
@sk_import pipeline:Pipeline

imb = pyimport("imblearn")
"""Lemaı̂tre G, Nogueira F, Aridas CK. Imbalanced-learn: A python toolbox to tackle the curse of imbalanced
datasets in machine learning. Journal of Machine Learning Research 18 (2017) 1–5."""

export brfc_prediction, cov_prediction, spectrum_prediction, risk_eval


"""
randomundersampling(
    trainclassPP,
    allTrain,
    numberoftimeframesTrain
    [; medicationT,
    meanperPatient = Vector{Hermitian},
    ratio = 0.6,
    split = 3]
)

Randomly undersample majority class (0!) by either splitting or a specified ratio.
The split stratisfies by medication, the ratio option does not.
In the paper, ratio=0.6 and split=3 was used for the covariance data, ratio=0.4 and
split=2 for the spectral data.
meanperPatient is only used for the covariance data, it holds the geometric mean
of all covariances for each patient.
numberoftimeframesTrain saves the number of timeframes per patient
(=number of covariances/spectra)

returns undersampled training set.
"""
function randomundersampling(
    trainclassPP,
    allTrain,
    numberoftimeframesTrain;
    medicationT = [],
    meanperPatient = Vector{Hermitian},
    ratio = 0.6,
    split = 3,
)
    trainclass = Vector(trainclassPP)
    Train = Vector{Vector{Hermitian}}(allTrain)
    ind0 = 1:length(trainclassPP)
    x = []
    if length(medicationT) > 2
        a = findall(iszero, trainclassPP[ind0])
        NoPODtrain = []
        NoPODtest = []
        kf = StratifiedKFold(n_splits = split, shuffle = true)

        for (train_index, test_index) in kf.split(a, medicationT[a])
            append!(NoPODtest, [test_index .+ 1])
            append!(NoPODtrain, [train_index .+ 1])
        end
        x = a[NoPODtrain[1]]
    else
        a = findall(iszero, trainclassPP[ind0])
        x = zeros(Int(round(ratio * length(a))))
        StatsBase.knuths_sample!(a, x)
    end

    x = unique!(sort!(Int.(x)))
    println(size(x))
    deleteat!(Train, x)
    deleteat!(trainclass, x)
    currentnumberoftimeframesTrain = Vector(numberoftimeframesTrain)
    deleteat!((currentnumberoftimeframesTrain), x)
    if length(meanperPatient) > 2
        mpP = Vector{Hermitian}(meanperPatient)
        deleteat!(mpP, x)
        return Train, trainclass, currentnumberoftimeframesTrain, mpP
    else
        return Train, trainclass, currentnumberoftimeframesTrain
    end
end

"""
riemaniandistanceoutlier(
    trainclass,
    Train,
    mpP,
    currentnumberoftimeframesTrain
    [; ratio_incl = 0.8,
    threshold = 0.85]
)

Filter out (some) outliers in the covariance training set, based on geometric
mean of each patient, using riemanian geometry.

returns filtered training set.
"""
function riemaniandistanceoutlier(
    trainclass,
    Train,
    mpP,
    currentnumberoftimeframesTrain;
    ratio_incl = 0.8,
    threshold = 0.85,
)
    ind0 = 1:length(trainclass)
    x = zeros(Int(round(ratio_incl * length(ind0))))
    StatsBase.knuths_sample!(ind0, x)
    x = unique!(sort!(Int.(x)))
    C = ℍVector(mpP[Int.(x)])
    gmsr1 = mean(
        Fisher,
        ℍVector([C[i] for i = 1:length(C) if trainclass[x[i]] < 0.5]),
    )
    gmsr2 = mean(
        Fisher,
        ℍVector([C[i] for i = 1:length(C) if trainclass[x[i]] > 0.5]),
    )
    potatomean = mean(Fisher, ℍVector([Hermitian(gmsr1), Hermitian(gmsr2)]))
    potatodist =
        getDistances(Fisher, ℍVector([Hermitian(potatomean)]), ℍVector(C))
    potatodist = reshape(potatodist, size(potatodist)[2])
    println(size(potatodist))
    quan = quantile!(potatodist, threshold)
    ppp = findall(potatodist .> quan)
    deleteat!(Train, ind0[ppp])
    deleteat!(trainclass, ind0[ppp])
    deleteat!((currentnumberoftimeframesTrain), ind0[ppp])
    deleteat!(mpP, ind0[ppp])
    return Train, trainclass, currentnumberoftimeframesTrain, mpP
end


"""
patientsplit(trainclass, currentnoTF)

Create the training set split for the grid search based on a split of patients.
(one could also use the GroupKFold)

returns splits
"""
function patientsplit(trainclass, currentnoTF)
    nops = Int(length(trainclass))
    X = 1:nops
    kf = StratifiedKFold(n_splits = 5, shuffle = true)
    kf.get_n_splits(X, trainclass)
    splits = []
    noTF = [sum(currentnoTF[1:i]) for i = 1:length(currentnoTF)]
    pushfirst!(noTF, 0)
    for (train_index, test_index) in kf.split(X, trainclass)
        full_trainindex =
            [noTF[i+1] + j for i in train_index for j = 0:currentnoTF[i+1]-1]
        full_testindex =
            [noTF[i+1] + j for i in test_index for j = 0:currentnoTF[i+1]-1]
        sp = (full_trainindex, full_testindex)
        append!(splits, [sp])
    end
    return splits
end

"""
brfc_prediction(train, trainclasses, test)

Interfaces imbalanced-learn BalancedRandomForestClassifier for the
classification of the patient data and patient+burstsupp data
to show the parameters used.

returns prediction on testset (p) and training set (pT).
"""
function brfc_prediction(train, trainclasses, test)
    m1 = ScikitLearn.fit!(
        imb.ensemble.BalancedRandomForestClassifier(
            max_depth = 4,
            n_estimators = 100,
            bootstrap = true,
            random_state = 0,
            #class_weight = Dict(1 => 1 / ratio),
            n_jobs = -1,
        ),
        train,
        trainclasses,
    )
    p = ScikitLearn.predict_proba(m1, test)[:, 2]
    pT = ScikitLearn.predict_proba(m1, train)[:, 2]
    return p, pT
end


"""
cov_prediction(
    Train,
    Test,
    trainclassPP,
    medication,
    medicationT,
    numberoftimeframesTest,
    numberoftimeframesTrain,
    meanperPatient,
    meanperTest,
    beta,
)

Classification for covariance time series data.
parameters:
Train  --> training set (of type Vector{Vector{Hermitian}})
Test --> test set (of type Vector{Vector{Hermitian}})
trainclassPP --> trainclasses
medication --> mediction of testset (not used at the moment, the function is called for the groups separately)
medicationT --> medication of training set (set to [] when only 1 medication is used)
numberoftimeframesTest --> number of covariances for each test patient (Vector)
numberoftimeframesTrain --> number of covariances for each training patient (Vector)
meanperPatient --> geometric mean for each training patient (of type Vector{Hermitian})
meanperTest --> geometric mean for each test patient (of type Vector{Hermitian}, not used)
beta --> balanced accuracy score threshold for training set to accept or reject a SVM
noS --> number of time frames for minority class
bags --> number of saved estimators

returns prediction on testset (p) and training set (pT).
"""
function cov_prediction(
    Train,
    Test,
    trainclassPP,
    medication,
    medicationT,
    numberoftimeframesTest,
    numberoftimeframesTrain,
    meanperPatient,
    meanperTest,
    beta,
    noS=15,
    bags=10,
)
    #------------------------------------------------------------------------------------------

    yTest = [0 for i = 1:length(Test) for j = 1:numberoftimeframesTest[i]]#zeros(size(Test)[1])#[
    yTrain =
        [0 for i = 1:length(trainclassPP) for j = 1:numberoftimeframesTrain[i]]
    allTrain = Vector{Vector{Hermitian}}(Train)
    r = 0
    out = 0
    while r < bags
        Train, trainclass, currentnumberoftimeframesTrain, mpP =
            randomundersampling(
                trainclassPP,
                allTrain,
                numberoftimeframesTrain;
                medicationT = medicationT,#leave out when there is only one med group
                meanperPatient = meanperPatient,
                ratio = 0.6,
                split = 3,
            )
        #this step (riemaniandistanceoutlier) was taken out for the final publication,
        #however it increases robustness,
        #especially when tranfering a model to an independent data set
        Train, trainclass, currentnumberoftimeframesTrain, mpP =
            riemaniandistanceoutlier(
                trainclass,
                Train,
                mpP,
                currentnumberoftimeframesTrain,
            )

        #sampling timeframes
        N_pos = sum(trainclass)
        N = length(trainclass)
        N_neg = N - N_pos
        ratio = N_neg / N_pos * 1
        noSpC = [noS, Int(ceil(noS * ratio))]
        println(noSpC)
        currentnoTF = [noSpC[trainclass[i]+1] for i = 1:length(trainclass)]
        train = Hermitian[
            Hermitian(Train[i][rand(1:currentnumberoftimeframesTrain[i])])
            for i = 1:length(trainclass) for j = 1:noSpC[trainclass[i]+1]
        ]
        trainclasses = [
            trainclass[i] for i = 1:length(trainclass) for
            j = 1:noSpC[trainclass[i]+1]
        ]

        alltrain = Hermitian[
            Hermitian(allTrain[i][j]) for i = 1:length(allTrain) for
            j = 1:numberoftimeframesTrain[i]
        ]
        trainclassall = [
            trainclassPP[i] for i = 1:length(allTrain) for
            j = 1:numberoftimeframesTrain[i]
        ]

        #calculate and project to tangent space
        train = tsMap(Fisher, train; meanISR = GMSR)
        alltrain = tsMap(Fisher, alltrain; meanISR = GMSR)
        test = tsMap(Fisher, test; meanISR = GMSR)

        splis = patientsplit(trainclass, currentnoTF)

        rfc = make_pipeline(
            StandardScaler(),
            SVC(
                class_weight = "balanced",
                random_state = 0,
                decision_function_shape = "ovo",
            ),
        )
        op = ScikitLearn.fit!(
            GridSearchCV(
                rfc,
                Dict(
                    "svc__C" => [0.25, 0.5, 0.75, 1, 2],
                    "svc__kernel" => ["rbf"],
                ),
                scoring = "roc_auc",
                n_jobs = -1,
                refit = true,
                cv = splits,
            ),
            train,
            trainclasses,
        )
        m2 = op.best_estimator_
        println(op.best_params_)
        println(op.best_score_)
        tmp = ScikitLearn.predict(m2, alltrain)

        yTrain = yTrain + tmp
        #check accuracy per patient on training set
        noTF = [
            sum(numberoftimeframesTrain[1:i]) for
            i = 1:length(numberoftimeframesTrain)
        ]
        pushfirst!(noTF, 0)
        y_pred = [
            Int(threshold.(mean(tmp[noTF[j]+1:noTF[j+1]]), 0.5)) for
            j = 1:length(noTF)-1
        ]
        alpha = balanced_accuracy_score(trainclassPP, y_pred)
        if alpha<beta
            println("out")
            println(mean(tmp))
            out=out+1
            if out>maximum([10, r*5])
                beta=beta-0.025
                println("change beta")
                println(out)
                println(beta)
                out=1
            end
            continue
        end
        tmp = ScikitLearn.predict(m2, test)
        yTest = tmp + yTest
        r = r + 1
    end
    #calculate per patient predictions
    yTrain = [yTrain[i] / bags for i = 1:length(yTrain)]
    noTF = [
        sum(numberoftimeframesTrain[1:i]) for
        i = 1:length(numberoftimeframesTrain)
    ]
    pushfirst!(noTF, 0)
    pT = [mean(yTrain[noTF[j]+1:noTF[j+1]]) for j = 1:length(noTF)-1]

    yTest = [yTest[i] / bags for i = 1:length(yTest)]
    println(mean(yTrain))
    println(mean(yTest))
    noTF = [
        sum(numberoftimeframesTest[1:i]) for
        i = 1:length(numberoftimeframesTest)
    ]
    pushfirst!(noTF, 0)

    p = [mean(yTest[noTF[j]+1:noTF[j+1]]) for j = 1:length(noTF)-1]
    return p, pT
end

"""
spectrum_prediction(
    Train,
    Test,
    trainclassPP,
    medication,
    medicationT,
    numberoftimeframesTest,
    numberoftimeframesTrain,
    beta,
)

Classification for spectral time series data.
parameters:
Train  --> training set (of type Vector{Vector{Hermitian}})
Test --> test set (of type Vector{Vector{Hermitian}})
trainclassPP --> trainclasses
medication --> mediction of testset (not used at the moment, the function is called for the groups separately)
medicationT --> medication of training set (set to [] when only 1 medication is used)
numberoftimeframesTest --> number of covariances for each test patient (Vector)
numberoftimeframesTrain --> number of covariances for each training patient (Vector)
beta --> balanced accuracy score threshold for training set to accept or reject a SVM
noS --> number of time frames for minority class
bags --> number of saved estimators

returns prediction on testset (p) and training set (pT).
"""
function spectrum_prediction(
    Train,
    Test,
    trainclassPP,
    medication,
    medicationT,
    numberoftimeframesTest,
    numberoftimeframesTrain,
    beta,
    bags = 15,
    noS = 5
)
    yTest = [0 for i = 1:length(Test) for j = 1:numberoftimeframesTest[i]]#zeros(size(Test)[1])#[
    yTrain =
        [0 for i = 1:length(trainclassPP) for j = 1:numberoftimeframesTrain[i]]
    allTrain = Vector{Vector{Matrix{Float64}}}(Train)
    r = 0
    out = 0
    while r < bags
        Train, trainclass, currentnumberoftimeframesTrain = randomundersampling(
            trainclassPP,
            allTrain,
            numberoftimeframesTrain;
            medicationT = medicationT,#leave out when there is only one med group
            ratio = 0.4,
            split = 2,
        )
        #sampling timeframes
        N_pos = sum(trainclass)
        N = length(trainclass)
        N_neg = N - N_pos
        ratio = N_neg / N_pos * 1
        noSpC = [noS, noS]
        currentnoTF = [noSpC[trainclass[i]+1] for i = 1:length(trainclass)]
        train = [
            reshape(
                mean(
                    Train[i][rand(1:currentnumberoftimeframesTrain[i])],
                    dims = 2,
                ),
                64,
            ) for i = 1:length(Train) for j = 1:noSpC[trainclass[i]+1]
        ]
        test = [
            reshape(mean(Test[i][j], dims = 2), 64) for i = 1:length(Test)
            for j = 1:numberoftimeframesTest[i]
        ]
        alltrain = [
            reshape(mean(allTrain[i][j], dims = 2), 64) for
            i = 1:length(allTrain) for j = 1:numberoftimeframesTrain[i]
        ]
        trainclasses = [
            trainclass[i] for i = 1:length(trainclass) for
            j = 1:noSpC[trainclass[i]+1]
        ]
        trainclassall = [
            trainclassPP[i] for i = 1:length(allTrain) for
            j = 1:numberoftimeframesTrain[i]
        ]

        splis = patientsplit(trainclass, currentnoTF)
        rfc = make_pipeline(
            StandardScaler(),
            SVC(
                class_weight = "balanced",
                random_state = 0,
                decision_function_shape = "ovo",
            ),
        )
        op = ScikitLearn.fit!(
            GridSearchCV(
                rfc,
                Dict(
                    "svc__C" => [0.25, 0.5, 0.75, 1, 3, 5, 10],
                    "svc__kernel" => ["rbf"],
                ),
                scoring = "roc_auc",
                n_jobs = -1,
                refit = true,
                cv = splits,
            ),
            train,
            trainclasses,
        )
        m2 = op.best_estimator_
        println(op.best_params_)
        println(op.best_score_)
        tmp = ScikitLearn.predict(m2, alltrain)
        yTrain = yTrain + tmp
        #check accuracy per patient on training set

        noTF = [
            sum(numberoftimeframesTrain[1:i]) for
            i = 1:length(numberoftimeframesTrain)
        ]
        pushfirst!(noTF, 0)
        y_pred = [
            Int(threshold.(mean(tmp[noTF[j]+1:noTF[j+1]]), 0.5)) for
            j = 1:length(noTF)-1
        ]
        alpha = balanced_accuracy_score(trainclassPP, y_pred)
        if alpha<beta
            println("out")
            println(mean(tmp))
            out=out+1
            if out>maximum([10, r*5])
                beta=beta-0.025
                println("change beta")
                println(out)
                println(beta)
                out=1
            end
            continue
        end

        tmp = ScikitLearn.predict(m2, test)
        noTF = [
            sum(numberoftimeframesTest[1:i]) for
            i = 1:length(numberoftimeframesTest)
        ]
        pushfirst!(noTF, 0)
        y_pred = [
            Int(round(mean(tmp[noTF[j]+1:noTF[j+1]]))) for j = 1:length(noTF)-1
        ]
        yTest = tmp + yTest
        r = r + 1
    end
    #calculate per patient predictions
    yTrain = [Int(round(yTrain[i] / bags)) for i = 1:length(yTrain)]
    noTF = [
        sum(numberoftimeframesTrain[1:i]) for
        i = 1:length(numberoftimeframesTrain)
    ]
    pushfirst!(noTF, 0)
    pT = [mean(yTrain[noTF[j]+1:noTF[j+1]]) for j = 1:length(noTF)-1]
    noTF = [
        sum(numberoftimeframesTest[1:i]) for
        i = 1:length(numberoftimeframesTest)
    ]
    pushfirst!(noTF, 0)

    yTest = [yTest[i] / bags for i = 1:length(yTest)]

    noTF = [
        sum(numberoftimeframesTest[1:i]) for
        i = 1:length(numberoftimeframesTest)
    ]
    pushfirst!(noTF, 0)

    p = [mean(yTest[noTF[j]+1:noTF[j+1]]) for j = 1:length(noTF)-1]
    return p, pT
end

"""
function risk_eval(trainclasses, pEEGT, pST, pMedT, pEEG, pS, pMed)

Risk evaluation taking probability of covariance prediction (pEEG[T]),
spectral data prectiction (pS[T]) and patient+burstsupp prediction (pMed[T])
into account. T marks the training set.

returns prediction on testset (pComb) and training set (pCombT).
"""
function risk_eval(trainclasses, pEEGT, pST, pMedT, pEEG, pS, pMed)
    add=[1]
    addT=[1]
    if roc_auc_score(trainclasses, pEEGT)>0.75 && roc_auc_score(trainclasses, pMedT)>0.75
        add = (pEEG+pMed)/2
        addT = (pEEGT+pMedT)/2
    elseif roc_auc_score(trainclasses, pST)>0.75 && roc_auc_score(trainclasses, pMedT)>0.75
        add = (pS+pMed)/2
        addT = (pST+pMedT)/2
    end
    if roc_auc_score(trainclasses, pEEGT)>0.75 && roc_auc_score(trainclasses, pMedT)>0.75 && roc_auc_score(trainclasses, pST)>0.75
        add = (pEEG+pS+pMed)/3
        addT = (pEEGT+pST+pMedT)/3
    end
    probneg=findall(x->x<0.25, add)
    probnegT=findall(x->x<0.25, addT)

    println("EEG THRESHOLD")
    fpr, tpr, threshold = roc_curve(trainclasses, pEEGT ,drop_intermediate=false)
    tpr[findall(tpr.<0.5)].=0
    gmean = sqrt.(tpr .*(1 .-fpr))
    println(argmax(gmean))
    println(gmean[argmax(gmean)])
    println(fpr[argmax(gmean)])
    println(tpr[argmax(gmean)])
    println(threshold[argmax(gmean)])
    eeg_th=min(threshold[argmax(gmean)], 1)

    println("Spectrum THRESHOLD")
    fpr, tpr, threshold = roc_curve(trainclasses, pST ,drop_intermediate=false)
    tpr[findall(tpr.<0.5)].=0
    gmean = sqrt.(tpr .*(1 .-fpr))
    println(argmax(gmean))
    println(gmean[argmax(gmean)])
    println(fpr[argmax(gmean)])
    println(tpr[argmax(gmean)])
    println(threshold[argmax(gmean)])
    spec_th=min(threshold[argmax(gmean)], 1)


    println("MED THRESHOLD")
    fpr, tpr, threshold = roc_curve(trainclasses, pMedT,drop_intermediate=false)
    tpr[findall(tpr.<0.5)].=0
    gmean = sqrt.(tpr .*(1 .-fpr))
    println(argmax(gmean))
    println(gmean[argmax(gmean)])
    println(fpr[argmax(gmean)])
    println(tpr[argmax(gmean)])
    println(threshold[argmax(gmean)])
    med_th=min(threshold[argmax(gmean)], 1)
    println("--------------")

    pComb = max.(min.(pEEG.+(0.5-eeg_th),pS.+(0.5-spec_th)),pMed.+(0.5-med_th))
    pCombT= max.(min.(pEEGT.+(0.5-eeg_th),pST.+(0.5-spec_th)),pMedT.+(0.5-eeg_th))
    println(roc_auc_score(trainclasses, pCombT))
    pCombTT=Array(pCombT)
    pCombTT[probnegT]=addT[probnegT]
    println(roc_auc_score(trainclasses, pCombTT))

    if roc_auc_score(trainclasses, pCombTT)>roc_auc_score(trainclasses, pCombT)
        println("Add")
        pCombT=pCombTT
        pComb[probneg]=add[probneg]
    else
        probneg=[]
        probnegT=[]
    end


    pCombTT= max.(min.(pEEGT.+(0.5-eeg_th), pST.+(0.5-spec_th)),pMedT)
    pCombTT[probnegT]=addT[probnegT]
    if roc_auc_score(trainclasses, pCombTT)>roc_auc_score(trainclasses, pCombT)
        println("EEG + th, Spec + th")
        pComb = max.(min.(pEEG.+(0.5-eeg_th), pS.+(0.5-spec_th)),pMed)
        pComb[probneg]=add[probneg]
        pCombT= pCombTT#max.(min.(pEEGT.+0.05,pST),pMedT)
        println(roc_auc_score(trainclasses, pCombT))
    end

    pCombTT= max.(min.(pEEGT.+(0.5-eeg_th), pST),pMedT.+(0.5-med_th))
    pCombTT[probnegT]=addT[probnegT]
    if roc_auc_score(trainclasses, pCombTT)>roc_auc_score(trainclasses, pCombT)
        println("EEG + th, Med + th")
        pComb = max.(min.(pEEG.+(0.5-eeg_th), pS),pMed.+(0.5-med_th))
        pComb[probneg]=add[probneg]
        pCombT= pCombTT#max.(min.(pEEGT.+0.05,pST),pMedT)
        println(roc_auc_score(trainclasses, pCombT))
    end

    pCombTT= max.(min.(pEEGT, pST.+(0.5-spec_th)),pMedT.+(0.5-med_th))
    pCombTT[probnegT]=addT[probnegT]
    if roc_auc_score(trainclasses, pCombTT)>roc_auc_score(trainclasses, pCombT)
        println("Spec + th, Med + th")
        pComb = max.(min.(pEEG, pS.+(0.5-spec_th)),pMed.+(0.5-med_th))
        pComb[probneg]=add[probneg]
        pCombT= pCombTT#max.(min.(pEEGT.+0.05,pST),pMedT)
        println(roc_auc_score(trainclasses, pCombT))
    end

    return pCombT, pComb
end
