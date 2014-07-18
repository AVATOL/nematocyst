function [ result ] = score_capsule_shape( allDataInstance, iter )
%SCORE_CAPSULE_SHAPE Summary of this function goes here
%   Detailed explanation goes here

% constants
BACKGROUND_LABEL = -1;
FOREGROUND_LABEL = 2; % 1 = basal, 2 = capsule

OUTPUT_FOLDER_NAME = 'Capsule Vis';

% don't do anything if image doesn't contain the foreground label
result = [];
if sum(double(allDataInstance.segLabels) == FOREGROUND_LABEL) == 0
    fprintf('skipping...\n');
    return;
end

% only necessary for visualization
label2color = containers.Map({-1, 0, 1, 2, 3}, {[64 64 64], [0 0 0], [0 255 0], [0 0 255], [255 0 0]});

% only get capsule
segLabels = allDataInstance.segLabels;
segLabels(segLabels ~= FOREGROUND_LABEL) = BACKGROUND_LABEL;

%% visualize
figure;
visualize_grid_image(allDataInstance.img, segLabels, label2color, allDataInstance.segs2);
print(gcf, '-dpng', sprintf('Data/%s/image_%d.png', OUTPUT_FOLDER_NAME, iter));
pause(0.1);
close;

%% convert image into binary mask for regionprops
labels = infer_pixels(segLabels, allDataInstance.segs2);
labels(labels == BACKGROUND_LABEL) = 0;
labels(labels == FOREGROUND_LABEL) = 1;

figure;
imshow(labels);
print(gcf, '-dpng', sprintf('Data/%s/mask_%d.png', OUTPUT_FOLDER_NAME, iter));
pause(0.01);
close;

%% get cc
cc = bwconncomp(labels);

%% get region props
result = regionprops(cc, 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength');

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