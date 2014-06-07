function visualize_annotations( rawDir, preprocessedDir, outputDir, label2color, splitsName, allData )
%VISUALIZE_ANNOTATIONS Visualize groundtruth folder.
%
%	rawDir:             folder path containing original images and annotations
%                           e.g. 'DataRaw/SomeDataset'
%	preprocessedDir:	folder path containing preprocessed data
%                           e.g. 'DataPreprocessed/SomeDataset'
%	outputDir:          folder path to output visualization
%                           e.g. 'ResultsPostprocessed/SomeExperiment'
%   label2color:        mapping from labels to colors (use containers.Map)
%   splitsName:         (optional) alternate name to splits folder
%	allData:            (optional) data structure containing all preprocessed data

narginchk(4, 6);

allDataAvailable = 0;
if nargin > 5
    allDataAvailable = 1;
end
if nargin < 5
    splitsName = 'splits';
end

rawDir = cleanup_path(rawDir);
preprocessedDir = cleanup_path(preprocessedDir);
outputDir = cleanup_path(outputDir);

%% constants
ANNOTATIONS_EXTENSION = '.jpg';

% %% label2color mapping from labels to colors
% label2color = containers.Map({0, 1, 2, 3, 4, 5, 6, 7, 8}, ...
%     {[0 0 0], [128 128 128], [128 128 0], [128 64 128], ...
%     [0 128 0], [0 0 128], [128 0 0], [128 80 0], ...
%     [255 128 0]});
% label2color = containers.Map({-1, 1}, {[64 64 64], [0 128 0]});

%% train, validation, test files
trainSplitsFile = [preprocessedDir filesep splitsName filesep 'Train.txt'];
fid = fopen(trainSplitsFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
testFiles = list{1,1};

validSplitsFile = [preprocessedDir filesep splitsName filesep 'Validation.txt'];
fid = fopen(validSplitsFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
testFiles = [testFiles; list{1,1}];

testSplitsFile = [preprocessedDir filesep splitsName filesep 'Test.txt'];
fid = fopen(testSplitsFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
testFiles = [testFiles; list{1,1}];

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% for each file
parfor f = 1:length(testFiles)
    fileName = testFiles{f};
    fprintf('On file %s...\n', fileName);

    %% read truth nodes
    if allDataAvailable
        truthLabels = allData{str2num(fileName)+1}.segLabels;
    else
        truthNodesPath = [preprocessedDir filesep 'nodes' filesep fileName '.txt'];
        [truthLabels, ~] = libsvmread(truthNodesPath);
    end

    %% read segments
    segmentsPath = [preprocessedDir filesep 'segments' filesep fileName '.txt'];
    segMat = dlmread(segmentsPath);

    %% read original image
    if allDataAvailable
        image =  allData{str2num(fileName)+1}.img;
    else
        originalImagePath = [rawDir filesep 'Images' filesep fileName ANNOTATIONS_EXTENSION];
        image = imread(originalImagePath);
    end

    %% visualize groundtruth
    truthImage = visualize_image(image, truthLabels, label2color, segMat);

    truthOutPath = [outputDir filesep fileName '.png'];
    imwrite(truthImage, truthOutPath);
end

end

function path = cleanup_path(path)

path = strrep(path, '/', filesep);
path = strrep(path, '\', filesep);

end
