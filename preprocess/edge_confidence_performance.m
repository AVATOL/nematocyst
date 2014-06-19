function [ positivePrecision, positiveRecall, negativePrecision, negativeRecall, positivePrecisionKL, positiveRecallKL, negativePrecisionKL, negativeRecallKL, model ] = edge_confidence_performance( allData, evalRange, trainRange, model, model_nodes )
%INITIAL_STATE_PERFORMANCE Summary of this function goes here
%   allData:	data structure containing all preprocessing data
%                   allData{i}.img mxnx3 uint8
%                   allData{i}.labels mxn double
%                   allData{i}.segs2 mxn double
%                   allData{i}.feat2 sxd double
%                   allData{i}.segLabels sx1 double
%                   allData{i}.adj sxs logical
%                   allData{i}.filename string (optional)
%                   allData{i}.segLocations sx2 double (optional)

DISPLAY = 1;

IGNORE_CLASSES = 0;

T_raw = 0.5;
T_label = 1;

%% train model
if nargin < 4
    fprintf('Model not specified; training model...\n');
    features = [];
    labels = [];
    
    for i = trainRange
        fprintf('\timage %d\n', i);
        
        [ai,aj,~] = find(allData{i}.adj);
        
        featDim = size(allData{i}.feat2, 2);
        edgeFeatures = zeros(length(ai), featDim);
        edgeLabels = zeros(length(ai), 1);
        for j = 1:length(ai)
            e1 = allData{i}.feat2(ai(j), :);
            e2 = allData{i}.feat2(aj(j), :);
            edgeFeatures(j, :) = abs(e1 - e2);
            edgeLabels(j, 1) = allData{i}.segLabels(ai(j), 1) == allData{i}.segLabels(aj(j), 1);
        end
        edgeLabels(edgeLabels == 0) = -1;
        
        features = [features; edgeFeatures];
        labels = [labels; edgeLabels];
    end
    features = sparse(features);
    fprintf('Training model...\n');
    model = train(labels, features, '-s 7 -c 10');
end

%% train model
if nargin < 5
    fprintf('Nodes model not specified; training model...\n');
    features = [];
    labels = [];
    
    for i = trainRange
        fprintf('\timage %d\n', i);
        features = [features; allData{i}.feat2];
        labels = [labels; allData{i}.segLabels];
    end
    features = sparse(features);
    fprintf('Training model...\n');
    model_nodes = train(labels, features, '-s 7 -c 10');
end

labelOrder = model.Label';

%% gather statistics
confidenceThresholds = 0:0.1:1;
positiveNumCorrect = zeros(1, length(confidenceThresholds));
positiveNumAbove = zeros(1, length(confidenceThresholds));
positiveNumTotal = zeros(1, length(confidenceThresholds));
negativeNumCorrect = zeros(1, length(confidenceThresholds));
negativeNumAbove = zeros(1, length(confidenceThresholds));
negativeNumTotal = zeros(1, length(confidenceThresholds));

positiveNumCorrectKL = zeros(1, length(confidenceThresholds));
positiveNumAboveKL = zeros(1, length(confidenceThresholds));
positiveNumTotalKL = zeros(1, length(confidenceThresholds));
negativeNumCorrectKL = zeros(1, length(confidenceThresholds));
negativeNumAboveKL = zeros(1, length(confidenceThresholds));
negativeNumTotalKL = zeros(1, length(confidenceThresholds));

positiveNumCorrectKL2 = zeros(1, length(confidenceThresholds));
positiveNumAboveKL2 = zeros(1, length(confidenceThresholds));
positiveNumTotalKL2 = zeros(1, length(confidenceThresholds));
negativeNumCorrectKL2 = zeros(1, length(confidenceThresholds));
negativeNumAboveKL2 = zeros(1, length(confidenceThresholds));
negativeNumTotalKL2 = zeros(1, length(confidenceThresholds));

for i = evalRange
    %% compute features and groundtruth
    [ai,aj,~] = find(allData{i}.adj);
    
    nodeFeatures = sparse(allData{i}.feat2);
    nodeGtLabels = allData{i}.segLabels;
    [node_predicted_label, node_accuracy, node_probs] = predict(nodeGtLabels, nodeFeatures, model, '-b 1');
    
    KLRawFeatsWeights = zeros(length(ai), 1);
    KLLabelDistWeights = zeros(length(ai), 1);
    
    featDim = size(allData{i}.feat2, 2);
    edgeFeatures = zeros(length(ai), featDim);
    edgeLabels = zeros(length(ai), 1);
    for j = 1:length(ai)
        e1 = allData{i}.feat2(ai(j), :);
        e2 = allData{i}.feat2(aj(j), :);
        edgeFeatures(j, :) = abs(e1 - e2);
        edgeLabels(j, 1) = allData{i}.segLabels(ai(j), 1) == allData{i}.segLabels(aj(j), 1);
        
        KLRawFeatsWeights(j, 1) = exp( -(mykldiv(e1, e2) + mykldiv(e2, e1))*T_raw/2 );
        
        l1 = node_probs(ai(j), :);
        l2 = node_probs(aj(j), :);
        KLLabelDistWeights(j, 1) = exp( -(mykldiv(l1, l2) + mykldiv(l2, l1))*T_label/2 );
    end
    edgeLabels(edgeLabels == 0) = -1;
    
    %% get probabilities
    features = edgeFeatures;
    gtLabels = edgeLabels;
    [predicted_label, accuracy, probs] = predict(gtLabels, sparse(features), model, '-b 1');
    
    %% for generating positive precision/recall results
    for tIndex = 1:length(confidenceThresholds)
        t = confidenceThresholds(tIndex);
        
        prediction = 1*ones(size(gtLabels));
        recalledGT = gtLabels;
        
        confidences = probs(:, 1);
        
        %% eliminate IGNORE CLASSES
        for ignoreClass = IGNORE_CLASSES
            ignoreIndices = find(recalledGT == ignoreClass);
            recalledGT(ignoreIndices) = [];
            prediction(ignoreIndices) = [];
            confidences(ignoreIndices) = [];
        end
        
        recalled = confidences >= t;
        recalledIndices = find(recalled == 0);
        prediction(recalledIndices) = [];
        recalledGT(recalledIndices) = [];
        
        corrects = prediction == recalledGT;

        %% compute
        positiveNumCorrect(1, tIndex) = positiveNumCorrect(1, tIndex) + sum(corrects == 1);
        positiveNumAbove(1, tIndex) = positiveNumAbove(1, tIndex) + numel(recalledGT);
        positiveNumTotal(1, tIndex) = positiveNumTotal(1, tIndex) + numel(confidences);
    end

    %% for generating positive precision/recall results
    for tIndex = 1:length(confidenceThresholds)
        t = confidenceThresholds(tIndex);
        
        prediction = -1*ones(size(gtLabels));
        recalledGT = gtLabels;
        
        confidences = probs(:, 1);
        
        %% eliminate IGNORE CLASSES
        for ignoreClass = IGNORE_CLASSES
            ignoreIndices = find(recalledGT == ignoreClass);
            recalledGT(ignoreIndices) = [];
            prediction(ignoreIndices) = [];
            confidences(ignoreIndices) = [];
        end
        
        recalled = confidences <= t;
        recalledIndices = find(recalled == 0);
        prediction(recalledIndices) = [];
        recalledGT(recalledIndices) = [];
        
        corrects = prediction == recalledGT;

        %% compute
        negativeNumCorrect(1, tIndex) = negativeNumCorrect(1, tIndex) + sum(corrects == 1);
        negativeNumAbove(1, tIndex) = negativeNumAbove(1, tIndex) + numel(recalledGT);
        negativeNumTotal(1, tIndex) = negativeNumTotal(1, tIndex) + numel(confidences);
    end
    
    %% for generating positive precision/recall results (KL)
    for tIndex = 1:length(confidenceThresholds)
        t = confidenceThresholds(tIndex);
        
        prediction = 1*ones(size(gtLabels));
        recalledGT = gtLabels;
        
        confidences = KLRawFeatsWeights;
        
        %% eliminate IGNORE CLASSES
        for ignoreClass = IGNORE_CLASSES
            ignoreIndices = find(recalledGT == ignoreClass);
            recalledGT(ignoreIndices) = [];
            prediction(ignoreIndices) = [];
            confidences(ignoreIndices) = [];
        end
        
        recalled = confidences >= t;
        recalledIndices = find(recalled == 0);
        prediction(recalledIndices) = [];
        recalledGT(recalledIndices) = [];
        
        corrects = prediction == recalledGT;

        %% compute
        positiveNumCorrectKL(1, tIndex) = positiveNumCorrectKL(1, tIndex) + sum(corrects == 1);
        positiveNumAboveKL(1, tIndex) = positiveNumAboveKL(1, tIndex) + numel(recalledGT);
        positiveNumTotalKL(1, tIndex) = positiveNumTotalKL(1, tIndex) + numel(confidences);
    end

    %% for generating positive precision/recall results (KL)
    for tIndex = 1:length(confidenceThresholds)
        t = confidenceThresholds(tIndex);
        
        prediction = -1*ones(size(gtLabels));
        recalledGT = gtLabels;
        
        confidences = KLRawFeatsWeights;
        
        %% eliminate IGNORE CLASSES
        for ignoreClass = IGNORE_CLASSES
            ignoreIndices = find(recalledGT == ignoreClass);
            recalledGT(ignoreIndices) = [];
            prediction(ignoreIndices) = [];
            confidences(ignoreIndices) = [];
        end
        
        recalled = confidences <= t;
        recalledIndices = find(recalled == 0);
        prediction(recalledIndices) = [];
        recalledGT(recalledIndices) = [];
        
        corrects = prediction == recalledGT;

        %% compute
        negativeNumCorrectKL(1, tIndex) = negativeNumCorrectKL(1, tIndex) + sum(corrects == 1);
        negativeNumAboveKL(1, tIndex) = negativeNumAboveKL(1, tIndex) + numel(recalledGT);
        negativeNumTotalKL(1, tIndex) = negativeNumTotalKL(1, tIndex) + numel(confidences);
    end
    
    %% for generating positive precision/recall results (KL label dist)
    for tIndex = 1:length(confidenceThresholds)
        t = confidenceThresholds(tIndex);
        
        prediction = 1*ones(size(gtLabels));
        recalledGT = gtLabels;
        
        confidences = KLLabelDistWeights;
        
        %% eliminate IGNORE CLASSES
        for ignoreClass = IGNORE_CLASSES
            ignoreIndices = find(recalledGT == ignoreClass);
            recalledGT(ignoreIndices) = [];
            prediction(ignoreIndices) = [];
            confidences(ignoreIndices) = [];
        end
        
        recalled = confidences >= t;
        recalledIndices = find(recalled == 0);
        prediction(recalledIndices) = [];
        recalledGT(recalledIndices) = [];
        
        corrects = prediction == recalledGT;

        %% compute
        positiveNumCorrectKL2(1, tIndex) = positiveNumCorrectKL2(1, tIndex) + sum(corrects == 1);
        positiveNumAboveKL2(1, tIndex) = positiveNumAboveKL2(1, tIndex) + numel(recalledGT);
        positiveNumTotalKL2(1, tIndex) = positiveNumTotalKL2(1, tIndex) + numel(confidences);
    end

    %% for generating positive precision/recall results (KL label dist)
    for tIndex = 1:length(confidenceThresholds)
        t = confidenceThresholds(tIndex);
        
        prediction = -1*ones(size(gtLabels));
        recalledGT = gtLabels;
        
        confidences = KLLabelDistWeights;
        
        %% eliminate IGNORE CLASSES
        for ignoreClass = IGNORE_CLASSES
            ignoreIndices = find(recalledGT == ignoreClass);
            recalledGT(ignoreIndices) = [];
            prediction(ignoreIndices) = [];
            confidences(ignoreIndices) = [];
        end
        
        recalled = confidences <= t;
        recalledIndices = find(recalled == 0);
        prediction(recalledIndices) = [];
        recalledGT(recalledIndices) = [];
        
        corrects = prediction == recalledGT;

        %% compute
        negativeNumCorrectKL2(1, tIndex) = negativeNumCorrectKL2(1, tIndex) + sum(corrects == 1);
        negativeNumAboveKL2(1, tIndex) = negativeNumAboveKL2(1, tIndex) + numel(recalledGT);
        negativeNumTotalKL2(1, tIndex) = negativeNumTotalKL2(1, tIndex) + numel(confidences);
    end
end

positivePrecision = positiveNumCorrect ./ positiveNumAbove;
positiveRecall = positiveNumAbove ./ positiveNumTotal;
negativePrecision = negativeNumCorrect ./ negativeNumAbove;
negativeRecall = negativeNumAbove ./ negativeNumTotal;

positivePrecisionKL = positiveNumCorrectKL ./ positiveNumAboveKL;
positiveRecallKL = positiveNumAboveKL ./ positiveNumTotalKL;
negativePrecisionKL = negativeNumCorrectKL ./ negativeNumAboveKL;
negativeRecallKL = negativeNumAboveKL ./ negativeNumTotalKL;

positivePrecisionKL2 = positiveNumCorrectKL2 ./ positiveNumAboveKL2;
positiveRecallKL2 = positiveNumAboveKL2 ./ positiveNumTotalKL2;
negativePrecisionKL2 = negativeNumCorrectKL2 ./ negativeNumAboveKL2;
negativeRecallKL2 = negativeNumAboveKL2 ./ negativeNumTotalKL2;

if DISPLAY ~= 0
    figure;
    subplot(1, 2, 1);
    plot(   confidenceThresholds, positivePrecision, 's--', ...
            confidenceThresholds, positivePrecisionKL, 's--', ...
            confidenceThresholds, positivePrecisionKL2, 's--');
    title('Superpixel precision for Positive Edge Threshold');
    xlabel('Threshold');
    ylabel('Precision');
    legend('Edge Classifier', sprintf('KL (Raw Feats), T=%.2f', T_raw), sprintf('KL (Label Dist), T=%.2f', T_label));

    subplot(1, 2, 2);
    plot(confidenceThresholds, positiveRecall, 's--', ...
            confidenceThresholds, positiveRecallKL, 's--', ...
            confidenceThresholds, positiveRecallKL2, 's--');
    title('Superpixel Recall for Positive Edge Threshold');
    xlabel('Threshold');
    ylabel('Recall');
    legend('Edge Classifier', sprintf('KL (Raw Feats), T=%.2f', T_raw), sprintf('KL (Label Dist), T=%.2f', T_label));
    
    figure;
    subplot(1, 2, 1);
    plot(confidenceThresholds, negativePrecision, 's--', ...
            confidenceThresholds, negativePrecisionKL, 's--', ...
            confidenceThresholds, negativePrecisionKL2, 's--');
    title('Superpixel precision for Negative Edge Threshold');
    xlabel('Threshold');
    ylabel('Precision');
    legend('Edge Classifier', sprintf('KL (Raw Feats), T=%.2f', T_raw), sprintf('KL (Label Dist), T=%.2f', T_label));

    subplot(1, 2, 2);
    plot(confidenceThresholds, negativeRecall, 's--', ...
            confidenceThresholds, negativeRecallKL, 's--', ...
            confidenceThresholds, negativeRecallKL2, 's--');
    title('Superpixel Recall for Negative Edge Threshold');
    xlabel('Threshold');
    ylabel('Recall');
    legend('Edge Classifier', sprintf('KL (Raw Feats), T=%.2f', T_raw), sprintf('KL (Label Dist), T=%.2f', T_label));
end

end

function val = mykldiv(p, q)

if length(p) ~= length(q)
    fprintf('dimensions not the same!\n');
end

val = 0;

pn = p ./ sum(p);
qn = q ./ sum(q);

for i = 1:length(p)
    if p(i) ~= 0
        if q(i) == 0
        else
            val = val + p(i) * log(p(i) / q(i));
        end
    end
end

end