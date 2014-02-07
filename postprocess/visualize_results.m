function visualize_results( rawDir, preprocessedDir, resultsDir, outputDir, timeRange, foldRange, splitsName, allData )
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
%   timeRange:          range of time bound
%   foldRange:          range of folds
%   splitsName:         (optional) alternate name to splits folder
%	allData:            (optional) data structure containing all preprocessed data

narginchk(6, 8);

allDataAvailable = 0;
if nargin > 7
    allDataAvailable = 1;
end
if nargin < 7
    splitsName = 'splits';
end

%% constants
ANNOTATIONS_EXTENSION = '.jpg';

%% label2color mapping from labels to colors
label2color = [255 0 0; 28 255 28; 56 56 255; 255 84 84; ...
    112 255 112; 140 140 255; 255 168 168; 0 196 196; ...
    224 0 224; 252 252 0];

%% search types
searchTypes = cell(1, 4);
searchTypes{1} = 'hc';
searchTypes{2} = 'hl';
searchTypes{3} = 'lc';
searchTypes{4} = 'll';

%% test files
testSplitsFile = [preprocessedDir '/' splitsName '/Test.txt'];
fid = fopen(testSplitsFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
testFiles = list{1,1};

%% for each fold
for fold = foldRange
    fprintf('On fold %d...\n', fold);
    
    %% for each search type
    for s = 1:length(searchTypes)
        searchType = searchTypes{s};
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
            truthImage = visualize_image(image, truthLabels, label2color, segMat);
            
            truthOutPath = sprintf('%s/%s/infer_%s_%s_test_time%d_fold%d.png', outputDir, searchType, fileName, searchType, 0, fold);
            imwrite(truthImage, truthOutPath);
            
            %% for each time step
            parfor timeStep = timeRange
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
                cutMat = dlmread(cutsPath);
                cutMat = spconvert(cutMat);
                cutMat = full(cutMat);

                nNodes = length(truthLabels);
                [h, w] = size(cutMat);
                cutMat = [cutMat zeros(h, nNodes-w)];
                [h, w] = size(cutMat);
                cutMat = [cutMat; zeros(nNodes-h, w)];
                
                %% visualize inference
                inferImage = visualize_image(image, labels, label2color, segMat, cutMat);
                
                inferOutPath = sprintf('%s/%s/infer_%s_%s_test_time%d_fold%d.png', outputDir, searchType, fileName, searchType, timeStep+1, fold);
                imwrite(inferImage, inferOutPath);
            end
        end
    end
end

end

