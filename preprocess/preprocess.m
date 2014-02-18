function [ allData ] = preprocess( imagesPath, labelsPath, splitsPath, outputPath )
%PREPROCESS Preprocesses folders of images and groundtruth labels into a
%data format for HCSearch to work. Performs feature extraction.
%
%This implementation creates a regular grid of HOG/SIFT patches.
%Therefore only need grayscale images.
%
%One can change the features (e.g. add color) by editing this file.
%
%	imagesPath:	folder path to images folder of *.jpg images
%                   e.g. 'DataRaw/SomeDataset/Images'
%	labelsPath:	folder path to groundtruth folder of *.jpg label masks
%                   e.g. 'DataRaw/SomeDataset/Annotations'
%	splitsPath:	folder path that contains Train.txt,
%               Validation.txt, Test.txt
%                   e.g. 'DataRaw/SomeDataset/Splits'
%	outputPath:	folder path to output preprocessed data
%                   e.g. 'DataPreprocessed/SomeDataset'
%
%	allData:	data structure containing all preprocessed data

%% argument checking
narginchk(4, 4);

%% parameters
PATCH_SIZE = 32; % size of patches
DESCRIPTOR = 1; % 0 = HOG, 1 = SIFT
RGB_2_GRAY = 0; % set to 1 if input images are RGB (need to be grayscale)
USE_BINARY = 1; % set to 1 if only extracting binary groundtruth

%% define label map - NEED TO EDIT BELOW TO MAP GROUNDTRUTH COLORS TO LABELS
% uses map container: index is (grayscale) color and value is label
if USE_BINARY
    labelMap = containers.Map({0, 255}, {-1, 1});
else
    labelMap = containers.Map({32 0 255 128 64}, {-1, 0, 1, 2, 3});
end

%% constants
TRAIN_LIST = 'Train.txt';
VALID_LIST = 'Validation.txt';
TEST_LIST = 'Test.txt';

IMAGE_EXT = '.jpg';
GROUNDTRUTH_EXT = '.jpg';

%% process list of training, validation, test splits
trainListFile = [splitsPath '/' TRAIN_LIST];
validListFile = [splitsPath '/' VALID_LIST];
testListFile = [splitsPath '/' TEST_LIST];

fid = fopen(trainListFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
trainList = list{1,1};
trainRange = 1:length(trainList);

fid = fopen(validListFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
validList = list{1,1};
validRange = length(trainList)+1:length(trainList)+length(validList);

fid = fopen(testListFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
testList = list{1,1};
testRange = length(trainList)+length(validList)+1:length(trainList)+length(validList)+length(testList);

fileArray = [trainList; validList; testList];

%% process images/groundtruth folder
nImages = length(fileArray);
allData = cell(1, nImages);
cnt = 1;
for i= 1:length(fileArray)
    file = fileArray{i, 1};
    
    %% get paths
    imgPath = [imagesPath '/' file IMAGE_EXT];
    gtPath = [labelsPath '/' file GROUNDTRUTH_EXT];
    fprintf('Processing image %s...\n', file);
    
    if ~exist(imgPath, 'file')
        error(['image "' imgPath '" does not exist']);
    end
    if ~exist(gtPath, 'file')
        error(['labels "' gtPath '" does not exist']);
    end
    
    %% get image
    img = imread(imgPath);
    if RGB_2_GRAY
        img = rgb2gray(img);
    end
    [img, height, width] = resize_image(img, PATCH_SIZE, PATCH_SIZE);
    
    %% extract features
    % TO CHANGE FEATURES, EDIT BELOW
    if DESCRIPTOR == 1
        featureMatrix = pre_extract_sift(img, PATCH_SIZE);
    elseif DESCRIPTOR == 0
        featureMatrix = pre_extract_hog(img, PATCH_SIZE);
    end
    [nodesHeight, nodesWidth, ~] = size(featureMatrix);
    
    %% get groundtruth
    labels = imread(gtPath);
    if RGB_2_GRAY
        labels = rgb2gray(img);
    end
    [labels, ~, ~] = resize_image(labels, PATCH_SIZE, PATCH_SIZE);
    
    %% extract labels
    [truthMatrix, labels] = pre_ground_truth(labels, PATCH_SIZE, labelMap);
    
    %% get segments and locations
    [segments, segLocations, segSizes] = getSegments(PATCH_SIZE, height, width);
    
    %% add to data structure
    allData{cnt}.img = img;
    allData{cnt}.labels = labels;
    allData{cnt}.segs2 = segments;
    allData{cnt}.feat2 = reshape(permute(featureMatrix, [2 1 3]), nodesWidth*nodesHeight, []);
    allData{cnt}.segLabels = reshape(permute(truthMatrix, [2 1]), nodesWidth*nodesHeight, []);
    allData{cnt}.adj = getAdjacencyMatrix(nodesHeight, nodesWidth);
    allData{cnt}.filename = file;
    allData{cnt}.segLocations = segLocations;
    allData{cnt}.segSizes = segSizes;
    
    cnt = cnt+1;
end

%% hand over to allData preprocessing
allData = preprocess_alldata(allData, outputPath, trainRange, validRange, testRange);

end

function [ segmentsMatrix, segLocations, segSizes ] = getSegments(patchSize, height, width)

NORMALIZE_LOCATIONS = 1;

segmentsMatrix = zeros(height, width);
segLocations = zeros(width*height/patchSize^2, 2);
segSizes = (patchSize*patchSize)*ones(width*height/patchSize^2, 1);
cnt = 1;
for row = 1:patchSize:height % important: row-major order
    for col = 1:patchSize:width
        ycomp = row:row+patchSize-1;
        xcomp = col:col+patchSize-1;
        
        segmentsMatrix(ycomp, xcomp) = cnt;
        
        x = (col+col+patchSize-1)/2;
        y = (row+row+patchSize-1)/2;
        
        if NORMALIZE_LOCATIONS
            x = x/width;
            y = y/height;
        else
            x = floor(x);
            y = floor(y);
        end
        
        segLocations(cnt, 1) = x;
        segLocations(cnt, 2) = y;
        
        cnt = cnt + 1;
    end
end

end

function [ adjMatrix ] = getAdjacencyMatrix(height, width)

% subtle note: row-major order
adjMatrix = false(width*height, width*height);

% set up left-right edges
for row = 0:height-1
    for col = 0:width-2
        ind1 = row*width + col;
        ind2 = ind1+1;
        adjMatrix(ind1+1, ind2+1) = 1;
        adjMatrix(ind2+1, ind1+1) = 1;
    end
end

% set up up-down edges
for row = 0:height-2
    for col = 0:width-1
        ind1 = row*width + col;
        ind2 = ind1+width;
        adjMatrix(ind1+1, ind2+1) = 1;
        adjMatrix(ind2+1, ind1+1) = 1;
    end
end

end