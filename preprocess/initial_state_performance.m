function [ meanVal, stdVal, accuracies, marginMeanVals, marginStdVals, marginAccuracies, model ] = initial_state_performance( allData, evalRange, trainRange, model )
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
nLabels = length(labelOrder);

%% gather statistics
allRanks = [];

correct = zeros(1, nLabels);
total = zeros(1, nLabels);


allMarginCounts = [];
cnt = 1;
marginThresholds = 0:0.1:1;
marginCorrect = zeros(1, length(marginThresholds));
marginTotal = zeros(1, length(marginThresholds));

for i = evalRange
    %% get probabilities
    features = sparse(allData{i}.feat2);
    gtLabels = allData{i}.segLabels;
    segments = allData{i}.segs2;
    [predicted_label, accuracy, probs] = predict(gtLabels, features, model, '-b 1');
    
    %% get the predicted labels ordered by confidence
    [sorted, indices] = sort(probs, 2, 'descend');
    predictedOrderedLabels = labelOrder(indices);
    
    %% compute rank of each predicted label
    ranksBinary = (predictedOrderedLabels == repmat(gtLabels, 1, size(predictedOrderedLabels, 2)));
    positions = ranksBinary .* repmat(1:size(ranksBinary, 2), size(ranksBinary, 1), 1);
    ranks = sum(positions, 2);
    
    allRanks = [allRanks; ranks];

    %% for generating rank graphs
    for r = 1:nLabels
        restrictedPredict = predictedOrderedLabels(:, 1:r);
        presence = (restrictedPredict == repmat(gtLabels, 1, size(restrictedPredict, 2)));
        presence = sum(presence, 2);
        segLabels = gtLabels .* presence + (gtLabels+1) .* (1-presence);

        %% ground truth pixel level
        pixelGT = allData{i}.labels;
        pixelGT = pixelGT(:);
        
        %% ground truth segment-level restricted to top R labels
        segGT = infer_pixels(segLabels, segments);
        segGT = segGT(:);

        %% eliminate IGNORE CLASSES
        for ignoreClass = IGNORE_CLASSES
            ignoreIndices = find(pixelGT == ignoreClass);
            pixelGT(ignoreIndices) = [];
            segGT(ignoreIndices) = [];
        end

        %% compute
        correct(1, r) = correct(1, r) + sum(sum(segGT == pixelGT));
        total(1, r) = total(1, r) + numel(pixelGT);
    end
    
    %% compute margin
    margins = abs(sorted - repmat(sorted(:, 1), 1, size(sorted, 2)));
    
    %% for generating rank graphs
    nSegments = size(gtLabels, 1);
    allMarginCounts = [allMarginCounts; zeros(nSegments, length(marginThresholds))];
    for tIndex = 1:length(marginThresholds);
        t = marginThresholds(tIndex);
        
        labelPositionsToKeep = margins <= t;
        restrictedPredict = predictedOrderedLabels .* labelPositionsToKeep + -314*(1-labelPositionsToKeep);
        presence = (restrictedPredict == repmat(gtLabels, 1, size(restrictedPredict, 2)));
        presence = sum(presence, 2);
        segLabels = gtLabels .* presence + (gtLabels+1) .* (1-presence);
        
        counts = sum(labelPositionsToKeep, 2);
        allMarginCounts(cnt:cnt+nSegments-1, tIndex) = counts;
        
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
        marginCorrect(1, tIndex) = marginCorrect(1, tIndex) + sum(sum(segGT == pixelGT));
        marginTotal(1, tIndex) = marginTotal(1, tIndex) + numel(pixelGT);
    end
    cnt = cnt + nSegments;
end

meanVal = mean(allRanks);
stdVal = std(allRanks);
accuracies = correct ./ total;
marginMeanVals = mean(allMarginCounts, 1);
marginStdVals = std(allMarginCounts, 0, 1);
marginAccuracies = marginCorrect ./ marginTotal;

plot(1:nLabels, accuracies, 's--');
title('Upper bound pixel accuracy for all ranks');
xlabel('Rank');
ylabel('Upper bound pixel accuracy');

pause;

plot(marginThresholds, marginAccuracies, 's--');
title('Upper bound pixel accuracy for all margin thresholds');
xlabel('Threshold');
ylabel('Upper bound pixel accuracy');

pause;

figure;
subplot(1,2,1);
plot(marginThresholds, marginMeanVals, 's--');
title('Average number of labels kept for all margin thresholds');
xlabel('Threshold');
ylabel('Average number of labels kept');

subplot(1,2,2);
plot(marginThresholds, marginStdVals, 's--');
title('Standard deviation of labels kept for all margin thresholds');
xlabel('Threshold');
ylabel('Standard deviation of labels kept');

pause;

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