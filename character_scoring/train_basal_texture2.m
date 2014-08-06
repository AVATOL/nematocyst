function [ svmModel, trainLabels, trainFeatures, annotations ] = train_basal_texture2( allData, siftDictionary, annotations )
%TRAIN_BASAL_TEXTURE2 Train model for basal tubule morphology

%% get svm model

USE_SAVED_ANNOTATIONS = nargin >= 3;

%% constants
BACKGROUND_LABEL = -1;
FOREGROUND_LABEL = 1; % 1 = basal, 2 = capsule

[~, dictSize] = size(siftDictionary);

%% settings
PATCH_SIZE = 32; % size of patches

% only necessary for visualization
label2color = containers.Map({-1, 0, 1, 2, 3}, {[64 64 64], [0 0 0], [0 255 0], [0 0 255], [255 0 0]});

trainFeatures = [];
trainLabels = [];
if ~USE_SAVED_ANNOTATIONS
    annotations = cell(length(allData), 1);
end

for index = 1:length(allData)
    fprintf('on %d...\n', index);
    allDataInstance = allData{index};

    % don't do anything if image doesn't contain the foreground label
    if sum(double(allDataInstance.segLabels) == FOREGROUND_LABEL) == 0
        fprintf('skipping %d...\n', index);
        continue;
    end
    
    % only get basal
    segLabels = allDataInstance.segLabels;
    segLabels(segLabels ~= FOREGROUND_LABEL) = BACKGROUND_LABEL;
    
    %% convert image into binary mask for regionprops - to get orientation
    labels = infer_pixels(segLabels, allDataInstance.segs2);
    labels(labels == BACKGROUND_LABEL) = 0;
    labels(labels == FOREGROUND_LABEL) = 1;

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

    gridLabels = reshape(allDataInstance.segLabels, [nPatchCols, nPatchRows])';
    nFeatures = size(allDataInstance.feat2, 2);
    gridFeatures = reshape(allDataInstance.feat2, [nPatchCols, nPatchRows, nFeatures]);
    gridFeatures = permute(gridFeatures, [2 1 3]);

    if useHorizontalScan == 1
        nPatches = nPatchCols;
    else
        nPatches = nPatchRows;
    end
    
    figure;
    
    if ~USE_SAVED_ANNOTATIONS
        annotations{index} = -1*ones(nPatches-1, 1);
    end
    for ind = 1:nPatches-1
        %% grab the two halves
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

        %% get relevant patches
        [x1, y1] = find(labelsPart1 == FOREGROUND_LABEL);
        if isempty(x1)
            continue;
        end
        
        [x2, y2] = find(labelsPart2 == FOREGROUND_LABEL);
        if isempty(x2)
            continue;
        end

        if ~USE_SAVED_ANNOTATIONS
            %% display visualization
            hold off;
            visualize_grid_image(allDataInstance.img, segLabels, label2color, allDataInstance.segs2);
            hold on;
            pos = ind*PATCH_SIZE;
            if useHorizontalScan == 1
                xline = [pos pos];
                yline = [1 size(allDataInstance.img, 1)];
            else
                xline = [1 size(allDataInstance.img, 2)];
                yline = [pos pos];
            end
            plot(xline,yline,'Color','y','LineStyle','-');

            %% ask user for label
            answer = questdlg('Correct split?');
            if strcmpi(answer, 'Yes')
                gtLabel = 1;
            elseif strcmpi(answer, 'No')
                gtLabel = 0;
            else
                error('premature exit');
            end

            annotations{index}(ind) = gtLabel;
        else
            gtLabel = annotations{index}(ind);
            fprintf('using label %d...\n', gtLabel);
        end
        
        %% get the patch features - use bag of SIFTs
        part1Representation = zeros(dictSize, 1);
        for i = 1:length(x1)
            feat = reshape(featuresPart1(x1(i), y1(i), :), [], 1);
            [~, k] = min(vl_alldist(feat, siftDictionary));
            part1Representation(k) = part1Representation(k) + 1;
        end

        part2Representation = zeros(dictSize, 1);
        for i = 1:length(x2)
            feat = reshape(featuresPart2(x2(i), y2(i), :), [], 1);
            [~, k] = min(vl_alldist(feat, siftDictionary));
            part2Representation(k) = part2Representation(k) + 1;
        end

        % single feature for training
        finalFeature = [part1Representation; part2Representation]';
        
        trainFeatures = [trainFeatures; finalFeature];
        trainLabels = [trainLabels; gtLabel];
    end
    
    close all;
end

%% train svm
svmModel = svmtrain(trainLabels, trainFeatures, '-b 1 -t 0 -c 1');

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