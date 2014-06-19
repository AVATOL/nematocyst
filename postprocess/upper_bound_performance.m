function [ acc ] = upper_bound_performance( allData, subset )
%UPPER_BOUND_PERFORMANCE Calculate upper bound pixel accuracy performance
%   allData:	data structure containing all preprocessing data
%                   allData{i}.img mxnx3 uint8
%                   allData{i}.labels mxn double
%                   allData{i}.segs2 mxn double
%                   allData{i}.feat2 sxd double
%                   allData{i}.segLabels sx1 double
%                   allData{i}.adj sxs logical
%                   allData{i}.filename string (optional)
%                   allData{i}.segLocations sx2 double (optional)
%   subset:     subset/range of allData to compute over
%   acc:        pixel accuracy

IGNORE_CLASSES = 0;

correct = 0;
total = 0;

for i = subset
    pixelGT = allData{i}.labels;
    segLabels = allData{i}.segLabels;
    segments = allData{i}.segs2;
    
    %% ground truth pixel-level
    pixelGT = pixelGT(:);
    
    %% ground truth segment-level
    segGT = infer_pixels(segLabels, segments);
    segGT = segGT(:);
    
    %% eliminate IGNORE CLASSES
    for ignoreClass = IGNORE_CLASSES
        ignoreIndices = find(pixelGT == ignoreClass);
        pixelGT(ignoreIndices) = [];
        segGT(ignoreIndices) = [];
    end
    
    %% compute
    correct = correct + sum(sum(segGT == pixelGT));
    total = total + numel(pixelGT);
end

acc = correct/total;

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