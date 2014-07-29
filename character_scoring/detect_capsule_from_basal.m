function [ output_args ] = detect_capsule_from_basal( allDataInstance, iter )
%DETECT_CAPSULE_FROM_BASAL Summary of this function goes here
%   Detailed explanation goes here

% constants
BACKGROUND_LABEL = -1;
BASAL_LABEL = 1; % 1 = basal, 2 = capsule
CAPSULE_LABEL = 2;

OUTPUT_FOLDER_NAME = 'capsule_test';
PATCH_SIZE = 32; % size of patches

% don't do anything if image doesn't contain the foreground label
if sum(double(allDataInstance.segLabels) == BASAL_LABEL) == 0 ...
        && sum(double(allDataInstance.segLabels) == CAPSULE_LABEL) == 0
    fprintf('skipping...\n');
    return;
end

if ~exist(['Data/' OUTPUT_FOLDER_NAME], 'dir')
    mkdir(['Data/' OUTPUT_FOLDER_NAME]);
end

% only necessary for visualization
label2color = containers.Map({-1, 0, 1, 2, 3}, {[64 64 64], [0 0 0], [0 255 0], [0 0 255], [255 0 0]});

% only get capsule
segLabels = allDataInstance.segLabels;
segLabels(segLabels ~= BASAL_LABEL) = BACKGROUND_LABEL;

% visualize
figure;
visualize_grid_image(allDataInstance.img, segLabels, label2color, allDataInstance.segs2);
print(gcf, '-dpng', sprintf('Data/%s/image_%d.png', OUTPUT_FOLDER_NAME, iter));
pause(0.1);

%% fit ellipse (or bounding box) to basal tubule

% convert image into binary mask for regionprops - to get orientation
labels = infer_pixels(segLabels, allDataInstance.segs2);
labels(labels == BACKGROUND_LABEL) = 0;
labels(labels == BASAL_LABEL) = 1;

figure;
imshow(labels);
print(gcf, '-dpng', sprintf('Data/%s/mask_%d.png', OUTPUT_FOLDER_NAME, iter));
pause(0.01);

% get orientation
cc = bwconncomp(labels);
result = regionprops(cc, 'Orientation');
orientation = result.Orientation;

%% rotate image based on orientation of basal tubule
rotatedImg = imrotate(allDataInstance.img, -orientation);

figure;
imshow(rotatedImg);
print(gcf, '-dpng', sprintf('Data/%s/canonical_%d.png', OUTPUT_FOLDER_NAME, iter));
pause(0.01);

%% determine constrained scanning window

%% scanning window

close all;

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