function [ truthMatrix, imageTruthMatrix ] = pre_ground_truth( imageTruth, patchSize, labelMap, USE_BINARY )
%PRE_GROUND_TRUTH Extract ground truth patches from ground truth image.
%   imageTruth:         ground truth image matrix (with colors)
%   patchSize:          patch size
%   labelMap:           mapping from label to grayscale color
%                       (index-1) is the label for the annotation value in image
%   truthMatrix:        matrix of ground truth patches
%   imageTruthMatrix:   ground truth image matrix (cleaned and with labels)

%% argument checking
narginchk(3, 4);

%% clean up
[ imageTruth, imageTruthMatrix ] = cleanup(imageTruth, labelMap);

%% settings
if nargin < 4
    USE_BINARY = 0;
end

%% setup
[height, width] = size(imageTruth);
labelColorSet = cell2mat(keys(labelMap));

%% process patches
labels = zeros(width*height/patchSize^2, 1);
count = 1;
for col = 1:patchSize:width
    for row = 1:patchSize:height
        ycomp = row:row+patchSize-1;
        xcomp = col:col+patchSize-1;

        % get the patch
        patchTruth = im2double(imageTruth(ycomp, xcomp));
        
        % accumulate count of each label in patch
        votes = zeros(length(labelColorSet), 1);
        for i = 1:length(votes)
            votes(i) = sum(sum(abs(patchTruth - labelColorSet(i)/255.0) < 0.001));
        end
        
        % select label with highest count
        [val, index] = max(votes);
        
        % if background is max and ties with another, don't use background
        if sum(votes(votes == val)) > 1 && index == 1
            new_votes = votes;
            new_votes(1) = 0;
            [~, index] = max(new_votes);
        end
        
        label = labelMap(labelColorSet(index));
        
        % assign label
        labels(count, 1) = label;
        count = count + 1;
    end
end

%% output
truthMatrix = reshape(labels, height/patchSize, width/patchSize);

end

function [ imageTruth, imageTruthMatrix ] = cleanup(imageTruth, labelMap)

imageTruthMatrix = zeros(size(imageTruth));

labelColorSet = cell2mat(keys(labelMap));
labelSet = cell2mat(values(labelMap));

[height, width] = size(imageTruth);
nLabels = length(labelColorSet);

labelColorDists = zeros(height, width, nLabels);

for i = 1:nLabels
    labelColorDists(:, :, i) = abs(labelColorSet(i) - double(imageTruth));
end
[~, indexMat] = min(labelColorDists, [], 3);
imageTruth = labelColorSet(indexMat);
imageTruthMatrix = labelSet(indexMat);

end