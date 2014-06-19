function [ meanVal, stdVal, accuracies, model ] = initial_state_performance( allData, evalRange, trainRange, model )
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
nLabels = length(labelOrder);

%% gather statistics
allRanks = [];

correct = zeros(1, nLabels);
total = zeros(1, nLabels);

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
end

meanVal = mean(allRanks);
stdVal = std(allRanks);
accuracies = correct ./ total;

if DISPLAY ~= 0
    figure;
    plot(1:nLabels, accuracies, 's--');
    title('Upper bound pixel accuracy for all ranks');
    xlabel('Rank');
    ylabel('Upper bound pixel accuracy');
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