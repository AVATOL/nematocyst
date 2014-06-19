function visualize_results( rawDir, preprocessedDir, resultsDir, outputDir, label2color, timeRange, foldRange, searchTypes, splitsName, allData )
%VISUALIZE_RESULTS Visualize results folder.
%
%	rawDir:             folder path containing original images and annotations
%                           e.g. 'DataRaw/SomeDataset'
%	preprocessedDir:	folder path containing preprocessed data
%                           e.g. 'DataPreprocessed/SomeDataset'
%	resultsDir:         folder path containing HC-Search results
%                           e.g. 'Results/SomeExperiment'
%	outputDir:          folder path to output visualization
%                           e.g. 'ResultsPostprocessed/SomeExperiment'
%   label2color:        mapping from labels to colors (use containers.Map)
%   timeRange:          range of time bound
%   foldRange:          range of folds
%   searchTypes:        list of search types 1 = HC, 2 = HL, 3 = LC, 4 = LL
%   splitsName:         (optional) alternate name to splits folder and file
%	allData:            (optional) data structure containing all preprocessed data

narginchk(7, 10);

allDataAvailable = 0;
if nargin > 9
    allDataAvailable = 1;
end
if nargin < 9
    splitsName = 'splits/Test.txt';
end
if nargin < 8
    searchTypes = [1 2 3 4];
end

USE_NEMATOCYST = 0;

%% constants
ANNOTATIONS_EXTENSION = '.jpg';

% %% label2color mapping from labels to colors
% label2color = containers.Map({0, 1, 2, 3, 4, 5, 6, 7, 8}, ...
%     {[0 0 0], [128 128 128], [128 128 0], [128 64 128], ...
%     [0 128 0], [0 0 128], [128 0 0], [128 80 0], ...
%     [255 128 0]});
% label2color = containers.Map({-1, 1}, {[64 64 64], [0 255 0]});

%% search types
searchTypesCollection = cell(1, 5);
searchTypesCollection{1} = 'hc';
searchTypesCollection{2} = 'hl';
searchTypesCollection{3} = 'lc';
searchTypesCollection{4} = 'll';
searchTypesCollection{5} = 'rl';

%% test files
testSplitsFile = [preprocessedDir '/' splitsName];
fid = fopen(testSplitsFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
testFiles = list{1,1};

%% for each fold
for fold = foldRange
    fprintf('On fold %d...\n', fold);
    
    %% for each search type
    for s = searchTypes
        searchType = searchTypesCollection{s};
        fprintf('On search type %s...\n', searchType);
        
        if ~exist([outputDir '/' searchType], 'dir')
            mkdir([outputDir '/' searchType]);
        end
        
        %% for each file
        for f = 1:length(testFiles)
            fileName = testFiles{f};
            fprintf('On file %s...\n', fileName);
            
            %% file stuff
            originalImagePath = [rawDir '/Images/' fileName ANNOTATIONS_EXTENSION];
            segmentsPath = [preprocessedDir '/segments/' fileName '.txt'];
            truthNodesPath = [preprocessedDir '/nodes/' fileName '.txt'];
            
            %% read truth nodes
            if allDataAvailable
                truthLabels = allData{str2num(fileName)+1}.segLabels;
            else
                [truthLabels, ~] = libsvmread(truthNodesPath);
            end
            
            %% read segments
            segMat = dlmread(segmentsPath);

            %% read original image
            if allDataAvailable
                image =  allData{str2num(fileName)+1}.img;
            else
                image = imread(originalImagePath);
            end
            
            %% visualize groundtruth
            truthOutPath = sprintf('%s/%s/infer_%s_%s_test_time%d_fold%d.png', outputDir, searchType, fileName, searchType, 0, fold);
            if ~USE_NEMATOCYST
                truthImage = visualize_image(image, truthLabels, label2color, segMat);
                imwrite(truthImage, truthOutPath);
            else
                visualize_grid_image(image, truthLabels, label2color, segMat);
                print(gcf, '-dpng', truthOutPath);
                close;
                pause(0.05);
            end
            
            %% for each time step
            %parfor timeStep = timeRange
            for timeStep = timeRange
                fprintf('On time step %d...\n', timeStep);
                edgesFileName = sprintf('edges_%s_test_time%d_fold%d_%s.txt', searchType, timeStep, fold, fileName);
                nodesFileName = sprintf('nodes_%s_test_time%d_fold%d_%s.txt', searchType, timeStep, fold, fileName);
                
                cutsPath = [resultsDir '/results/' edgesFileName];
                nodesPath = [resultsDir '/results/' nodesFileName];

                if ~exist(cutsPath, 'file') || ~exist(nodesPath, 'file')
                    continue;
                end
                
                %% read nodes
                [labels, ~] = libsvmread(nodesPath);
                
                %% read edges
                cutFile = dir(cutsPath);
                if cutFile.bytes == 0
                    %% visualize inference
                    if ~USE_NEMATOCYST
                        inferImage = visualize_grid_image(image, labels, label2color, segMat, cutMat);
                    else
                        visualize_grid_image(image, labels, label2color, segMat);
                    end
                else
                    cutMat = dlmread(cutsPath);
                    cutMat = spconvert(cutMat);
                    cutMat = full(cutMat);

                    nNodes = length(truthLabels);
                    [h, w] = size(cutMat);
                    cutMat = [cutMat zeros(h, nNodes-w)];
                    [h, w] = size(cutMat);
                    cutMat = [cutMat; zeros(nNodes-h, w)];

                    %% visualize inference
                    if ~USE_NEMATOCYST
                        inferImage = visualize_grid_image(image, labels, label2color, segMat, cutMat);
                    else
                        visualize_grid_image(image, labels, label2color, segMat, cutMat);
                    end
                end
                
                inferOutPath = sprintf('%s/%s/infer_%s_%s_test_time%d_fold%d.png', outputDir, searchType, fileName, searchType, timeStep+1, fold);
                if ~USE_NEMATOCYST
                    imwrite(inferImage, inferOutPath);
                else
                    print(gcf, '-dpng', inferOutPath);
                    close;
                    pause(0.05);
                end
            end
        end
    end
end

end

