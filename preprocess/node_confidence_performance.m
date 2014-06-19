function [ margin3Precision, margin3Recall, margin2Accuracies, model ] = node_confidence_performance( allData, evalRange, trainRange, model )
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

%% train model
if nargin < 4
    fprintf('Model not specified; training model...\n');
    features = [];
    labels = [];
    
    for i = trainRange
        fprintf('\timage %d\n', i);
        features = [features; allData{i}.feat2];
        labels = [labels; allData{i}.segLabels];
    end
    features = sparse(features);
    fprintf('Training model...\n');
    model = train(labels, features, '-s 7 -c 10');
end

labelOrder = model.Label';

%% gather statistics
marginThresholds = 0:0.1:1;
margin2Correct = zeros(1, length(marginThresholds));
margin2Total = zeros(1, length(marginThresholds));
margin3Correct = zeros(1, length(marginThresholds));
margin3Above = zeros(1, length(marginThresholds));
margin3Total = zeros(1, length(marginThresholds));

for i = evalRange
    %% get probabilities
    features = sparse(allData{i}.feat2);
    gtLabels = allData{i}.segLabels;
    segments = allData{i}.segs2;
    [predicted_label, accuracy, probs] = predict(gtLabels, features, model, '-b 1');
    
    %% get the predicted labels ordered by confidence
    [sorted, indices] = sort(probs, 2, 'descend');
    predictedOrderedLabels = labelOrder(indices);
    
    %% for generating threshold results
    for tIndex = 1:length(marginThresholds)
        t = marginThresholds(tIndex);
        labelPositionsIfConfident = [ones(size(sorted, 1), 1) zeros(size(sorted, 1), size(sorted, 2)-1)];
        labelPositionsIfNotConfident = [ones(size(sorted, 1), 4) zeros(size(sorted, 1), size(sorted, 2)-4)];
        labelPositionsToKeep = repmat((sorted(:, 1) >= t), 1, size(sorted, 2)) .* labelPositionsIfConfident ...
            + repmat((sorted(:, 1) < t), 1, size(sorted, 2)) .* labelPositionsIfNotConfident;
        
        restrictedPredict = predictedOrderedLabels .* labelPositionsToKeep + -314*(1-labelPositionsToKeep);
        presence = (restrictedPredict == repmat(gtLabels, 1, size(restrictedPredict, 2)));
        presence = sum(presence, 2);
        segLabels = gtLabels .* presence + (gtLabels+1) .* (1-presence);
        
        %% ground truth pixel level
        pixelGT = allData{i}.labels;
        pixelGT = pixelGT(:);
        
        %% ground truth segment-level restricted to margin threshold
        segGT = infer_pixels(segLabels, segments);
        segGT = segGT(:);

        %% eliminate IGNORE CLASSES
        for ignoreClass = IGNORE_CLASSES
            ignoreIndices = find(pixelGT == ignoreClass);
            pixelGT(ignoreIndices) = [];
            segGT(ignoreIndices) = [];
        end

        %% compute
        margin2Correct(1, tIndex) = margin2Correct(1, tIndex) + sum(sum(segGT == pixelGT));
        margin2Total(1, tIndex) = margin2Total(1, tIndex) + numel(pixelGT);
    end
    
    %% for generating precision/recall results
    for tIndex = 1:length(marginThresholds)
        t = marginThresholds(tIndex);
        
        prediction = predictedOrderedLabels(:, 1);
        recalledGT = gtLabels;
        
        confidences = sorted(:, 1);
        
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
        margin3Correct(1, tIndex) = margin3Correct(1, tIndex) + sum(corrects == 1);
        margin3Above(1, tIndex) = margin3Above(1, tIndex) + numel(recalledGT);
        margin3Total(1, tIndex) = margin3Total(1, tIndex) + numel(confidences);
    end
end

margin3Precision = margin3Correct ./ margin3Above;
margin3Recall = margin3Above ./ margin3Total;
margin2Accuracies = margin2Correct ./ margin2Total;

if DISPLAY ~= 0
    figure;
    plot(marginThresholds, margin2Accuracies, 's--');
    title('Upper bound pixel accuracy for confidence thresholds');
    xlabel('Threshold');
    ylabel('Upper bound pixel accuracy');

    figure;
    subplot(1, 2, 1);
    plot(marginThresholds, margin3Precision, 's--');
    title('Superpixel precision');
    xlabel('Threshold');
    ylabel('Precision');

    subplot(1, 2, 2);
    plot(marginThresholds, margin3Recall, 's--');
    title('Superpixel Recall');
    xlabel('Threshold');
    ylabel('Recall');
end

end

function [inferPixels] = infer_pixels(inferLabels, segments)

inferPixels = zeros(size(segments));
nNodes = length(inferLabels);

for i = 1:nNodes
    temp = segments;
    temp(temp ~= i) = 0;
    temp(temp == i) = inferLabels(i);
    
    inferPixels = inferPixels + temp;
end

end