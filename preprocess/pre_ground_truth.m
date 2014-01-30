function [ truthMatrix ] = pre_ground_truth( imageTruth, patchSize, labelMap, USE_BINARY )
%PRE_GROUND_TRUTH Extract ground truth patches from ground truth image.
%   imageTruth:     ground truth image matrix
%   patchSize:      patch size
%   labelMap:       mapping from label to grayscale color
%                   (index-1) is the label for the annotation value in image
%   truthMatrix:    matrix of ground truth patches

%% argument checking
narginchk(3, 4);

%% settings
if nargin < 4
    USE_BINARY = 0;
end

%% setup
[M, N] = size(imageTruth);

%% process patches
labels = zeros(N*M/patchSize^2, 1);
count = 1;
for col = 1:patchSize:N
    for row = 1:patchSize:M
        ycomp = row:row+patchSize-1;
        xcomp = col:col+patchSize-1;

        % get the patch
        patchTruth = im2double(imageTruth(ycomp, xcomp));
        
        % accumulate count of each label in patch
        votes = zeros(length(labelMap), 1);
        for i = 1:length(votes)
            votes(i) = sum(sum(abs(patchTruth - labelMap(i)/255.0) < 0.01));
        end
        
        % select label with highest count
        [val, index] = max(votes);
        
        % if background is max and ties with another, don't use background
        if sum(votes(votes == val)) > 1 && index == 1
            new_votes = votes;
            new_votes(1) = 0;
            [~, index] = max(new_votes);
        end
        
        if USE_BINARY
            label = index-1;
        else
            label = index-2;
        end
        
        % assign label
        labels(count, 1) = label;
        count = count + 1;
    end
end

if USE_BINARY
    labels(labels == 0) = -1;  % makes {-1,1} binary labels
end

%% output
truthMatrix = reshape(labels, M/patchSize, N/patchSize);

end

