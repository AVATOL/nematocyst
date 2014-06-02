function visualize_all_edge_weights( rawDir, preprocessedDir, outputDir, label2color, splitsName, allData )
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

narginchk(4, 6);

allDataAvailable = 0;
if nargin > 5
    allDataAvailable = 1;
end
if nargin < 5
    splitsName = 'splits/Test.txt';
end

%% constants
ANNOTATIONS_EXTENSION = '.jpg';

% %% label2color mapping from labels to colors
% label2color = containers.Map({0, 1, 2, 3, 4, 5, 6, 7, 8}, ...
%     {[0 0 0], [128 128 128], [128 128 0], [128 64 128], ...
%     [0 128 0], [0 0 128], [128 0 0], [128 80 0], ...
%     [255 128 0]});
% label2color = containers.Map({-1, 1}, {[64 64 64], [0 255 0]});

%% test files
testSplitsFile = [preprocessedDir '/' splitsName];
fid = fopen(testSplitsFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
testFiles = list{1,1};

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% for each file
parfor f = 1:length(testFiles)
    fileName = testFiles{f};
    fprintf('On file %s...\n', fileName);

    %% file stuff
    originalImagePath = [rawDir '/Images/' fileName ANNOTATIONS_EXTENSION];
    segmentsPath = [preprocessedDir '/segments/' fileName '.txt'];
    truthNodesPath = [preprocessedDir '/nodes/' fileName '.txt'];
    edgeWeightsPath = [preprocessedDir '/edgeweights/' fileName '.txt'];

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
    
    %% read edge weights
    wAdjMat = dlmread(edgeWeightsPath);
    wAdjMat = spconvert(wAdjMat);
    wAdjMat = full(wAdjMat);
    
    nNodes = length(truthLabels);
    [h, w] = size(wAdjMat);
    wAdjMat = [wAdjMat zeros(h, nNodes-w)];
    [h, w] = size(wAdjMat);
    wAdjMat = [wAdjMat; zeros(nNodes-h, w)];
    
    %% visualize groundtruth
    truthOutPath = sprintf('%s/%s.png', outputDir, fileName);
    truthImage = visualize_edge_weights(image, truthLabels, label2color, segMat, wAdjMat);
    imwrite(truthImage, truthOutPath);
end

end

