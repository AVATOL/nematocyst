function [ truthMatrix, imageTruthMatrix ] = pre_ground_truth( imageTruth, patchSize, labelMap )
%PRE_GROUND_TRUTH Extract ground truth patches from ground truth image.
%   imageTruth:         ground truth image matrix (with colors)
%   patchSize:          patch size
%   labelMap:           mapping from label to grayscale color
%                       (index-1) is the label for the annotation value in image
%   truthMatrix:        matrix of ground truth patches
%   imageTruthMatrix:   ground truth image matrix (cleaned and with labels)

%% argument checking
narginchk(3, 3);

%% clean up
[ ~, imageTruthMatrix ] = cleanup(imageTruth, labelMap);

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
        patchTruth = imageTruth(ycomp, xcomp);
        
        % accumulate count of each label in patch
        votes = zeros(length(labelColorSet), 1);
        for i = 1:length(votes)
            votes(i) = sum(sum(double(abs(labelColorSet(i)/255.0 - double(patchTruth)/255.0) < 0.025)));
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

labelColorSet = cell2mat(keys(labelMap));
labelSet = cell2mat(values(labelMap));

[height, width] = size(imageTruth);
nLabels = length(labelColorSet);

labelColorDists = zeros(height, width, nLabels);

for i = 1:nLabels
    labelColorDists(:, :, i) = double(abs(labelColorSet(i)/255.0 - double(imageTruth)/255.0) < 0.025);
end
[~, indexMat] = max(labelColorDists, [], 3);
imageTruth = labelColorSet(indexMat);
imageTruthMatrix = labelSet(indexMat);

end