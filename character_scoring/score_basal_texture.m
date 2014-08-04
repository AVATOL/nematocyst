function [ charState ] = score_basal_texture( allDataInstance )
%SCORE_BASAL_TEXTURE Score basal tubule morphology
%
%   allDataInstance:    image and labeling data
%   charState:          character state
%                           0 = homogeneous
%                           1 = heterogeneous

%% constants
BACKGROUND_LABEL = -1;
FOREGROUND_LABEL = 1; % 1 = basal, 2 = capsule

CHAR_STATE_HOMOGENEOUS = '0';
CHAR_STATE_HETEROGENEOUS = '1';

%% settings
HETEROGENEOUS_THRESHOLD = 0; % 0 = if one mode is sufficient

% OUTPUT_FOLDER_NAME = 'basal_test';
PATCH_SIZE = 32; % size of patches

% don't do anything if image doesn't contain the foreground label
if sum(double(allDataInstance.inferLabels) == FOREGROUND_LABEL) == 0
    fprintf('skipping...\n');
    return;
end

% if ~exist(['Data/' OUTPUT_FOLDER_NAME], 'dir')
%     mkdir(['Data/' OUTPUT_FOLDER_NAME]);
% end

% % only necessary for visualization
% label2color = containers.Map({-1, 0, 1, 2, 3}, {[64 64 64], [0 0 0], [0 255 0], [0 0 255], [255 0 0]});

% only get capsule
segLabels = allDataInstance.inferLabels;
segLabels(segLabels ~= FOREGROUND_LABEL) = BACKGROUND_LABEL;

%% visualize
% figure;
% visualize_grid_image(allDataInstance.img, segLabels, label2color, allDataInstance.segs2);
% print(gcf, '-dpng', sprintf('Data/%s/image_%d.png', OUTPUT_FOLDER_NAME, iter));
% pause(0.1);

%% convert image into binary mask for regionprops - to get orientation
labels = infer_pixels(segLabels, allDataInstance.segs2);
labels(labels == BACKGROUND_LABEL) = 0;
labels(labels == FOREGROUND_LABEL) = 1;

% figure;
% imshow(labels);
% print(gcf, '-dpng', sprintf('Data/%s/mask_%d.png', OUTPUT_FOLDER_NAME, iter));
% pause(0.01);

%% get skeleton
% skeleton = bwmorph(labels, 'thin', Inf);

% figure;
% imshow(skeleton);
% print(gcf, '-dpng', sprintf('Data/%s/thin_%d.png', OUTPUT_FOLDER_NAME, iter));
% pause(0.01);

%% get cc
cc = bwconncomp(labels);

%% get region props
result = regionprops(cc, 'Orientation');
useHorizontalScan = 1;
if length(result) == 1
    orientation = result.Orientation;
    fprintf('orientation=%f\n', orientation);
    if orientation > 45 || orientation < -45
        useHorizontalScan = 0;
    end
end
fprintf('using horizontal scan=%d\n', useHorizontalScan);

%% perform scan to find variance differences
[height, width] = size(allDataInstance.img);
nPatchRows = height / PATCH_SIZE;
nPatchCols = width / PATCH_SIZE;

gridLabels = reshape(allDataInstance.inferLabels, [nPatchCols, nPatchRows])';
nFeatures = size(allDataInstance.feat2, 2);
gridFeatures = reshape(allDataInstance.feat2, [nPatchCols, nPatchRows, nFeatures]);
gridFeatures = permute(gridFeatures, [2 1 3]);

if useHorizontalScan == 1
    nPatches = nPatchCols;
else
    nPatches = nPatchRows;
end

variances = zeros(nPatches-1, 1);
for ind = 1:nPatches-1
    % grab the two halves
    if useHorizontalScan == 1
        labelsPart1 = gridLabels(:, 1:ind);
        labelsPart2 = gridLabels(:, ind+1:end);

        featuresPart1 = gridFeatures(:, 1:ind, :);
        featuresPart2 = gridFeatures(:, ind+1:end, :);
    else
        labelsPart1 = gridLabels(1:ind, :);
        labelsPart2 = gridLabels(ind+1:end, :);

        featuresPart1 = gridFeatures(1:ind, :, :);
        featuresPart2 = gridFeatures(ind+1:end, :, :);
    end
    
    % get the patch features
    [x, y] = find(labelsPart1 == FOREGROUND_LABEL);
    part1Patches = [];
    for i = 1:length(x)
        part1Patches = [part1Patches; reshape(featuresPart1(x(i), y(i), :), 1, [])];
    end
    [x, y] = find(labelsPart2 == FOREGROUND_LABEL);
    part2Patches = [];
    for i = 1:length(x)
        part2Patches = [part2Patches; reshape(featuresPart2(x(i), y(i), :), 1, [])];
    end

    % calculate variances
    if size(part1Patches, 1) <= 1
        variances(ind) = inf;
    elseif size(part2Patches, 1) <= 1
        variances(ind) = inf;
    else
        part1var = cov(part1Patches);
        part2var = cov(part2Patches);
        
        % calculate difference
        variances(ind) = norm(sparse(part1var - part2var), 'fro')/(norm(sparse(part1var), 'fro')*norm(sparse(part2var), 'fro'));
    end
end

%% find the modes
segments = watershed(variances);
modeLocations = segments == 0;
% modes = modeLocations .* variances;
% modes(isnan(modes)) = 0;
% modes(modes == 0) = Inf;

%% plot
% figure;
% plot(1:nPatches-1, variances, 'x-');
% hold on;
% plot(1:nPatches-1, modes, 'ro', 'MarkerSize', 15, 'LineWidth', 2);
% if useHorizontalScan == 1
%     xlabel('x-coordinate Patch Position Threshold');
% else
%     xlabel('y-coordinate Patch Position Threshold');
% end
% ylabel('Normalized Difference of Covariances of Two Regions');
% title('Determination of 1/2-Kind Basal Tubule');
% grid on;
% print(gcf, '-dpng', sprintf('Data/%s/plot_%d.png', OUTPUT_FOLDER_NAME, iter));
% pause(0.1);
% 
% close all;

%% score character
charState = CHAR_STATE_HOMOGENEOUS;
if sum(double(modeLocations)) > HETEROGENEOUS_THRESHOLD
    charState = CHAR_STATE_HETEROGENEOUS;
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