function [ allData ] = preprocess_avatol( basePath, trainingList, charID, color2label, outputPath )
%PREPROCESS_AVATOL Preprocesses training examples from the AVATOL system 
%into a data format for HCSearch to work. Performs feature extraction. 
%This implementation creates a regular grid of HOG/SIFT patches on
%grayscale images.
%
%One can change the features (e.g. add color) by editing this file.
%
%   basePath:       base path to dataset
%   trainingList:   cell of structs where a struct denotes 
%                   a training instance and each struct has
%                   	.pathToMedia: path to image
%                   	.charState: character state
%                   	.pathToAnnotation: path to annotation
%                   	.taxonID: taxon ID
%   charID:         character ID string
%   color2label:    mapping from groundtruth colors to label (use containers.Map)
%	outputPath:     folder path to output preprocessed data
%                   e.g. 'DataPreprocessed/SomeDataset'
%
%	allData:        data structure containing all preprocessed data

%% argument checking
narginchk(5, 5);

%% parameters - tune if necessary
PATCH_SIZE = 32; % size of patches
DESCRIPTOR = 1; % 0 = HOG, 1 = SIFT

%% process images/groundtruth folder
nImages = length(trainingList);
allData = cell(1, nImages);
cnt = 1;
for i = 1:nImages
    imagesPath = normalize_file_sep([basePath filesep trainingList{i}.pathToMedia]);
    annotationPath = normalize_file_sep([basePath filesep trainingList{i}.pathToAnnotation]);
    
    %% get paths
    fprintf('Processing image %i...\n', i);
    
    if ~exist(imagesPath, 'file')
        error(['image "' imagesPath '" does not exist']);
    end
    if ~exist(annotationPath, 'file')
        error(['annotations "' annotationPath '" does not exist']);
    end
    
    %% get image
    img = imread(imagesPath);
    if ndims(img) == 3
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
    
    %% get groundtruth - read polygon data and get mask
    objects = read_annotation_file(annotationPath, charID);
    labels = polygons2masks(img, objects);
    [labels, ~, ~] = resize_image(labels, PATCH_SIZE, PATCH_SIZE);
    
    %% extract labels
    [truthMatrix, labels] = pre_ground_truth(labels, PATCH_SIZE, color2label);
    
    %% get segments and locations
    [segments, segLocations, segSizes] = getSegments(PATCH_SIZE, height, width);
    
    %% add to data structure
    allData{cnt}.img = img;
    allData{cnt}.labels = labels;
    allData{cnt}.segs2 = segments;
    allData{cnt}.feat2 = reshape(permute(featureMatrix, [2 1 3]), nodesWidth*nodesHeight, []);
    allData{cnt}.segLabels = reshape(permute(truthMatrix, [2 1]), nodesWidth*nodesHeight, []);
    allData{cnt}.adj = getAdjacencyMatrix(nodesHeight, nodesWidth);
    allData{cnt}.segLocations = segLocations;
    allData{cnt}.segSizes = segSizes;
    allData{cnt}.avatol = trainingList{i};
    
    cnt = cnt+1;
end

%% hand over to allData preprocessing
allData = preprocess_alldata(allData, outputPath, 1:nImages, [], []);

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

function objects = read_annotation_file(annotationFilePath, charID)
% parses annotation objects from file
% format: x1,y1;...;xn,yn:charID:charState

%% constants
DATA_DELIMITER = ':';
POINTS_DELIMITER = ';';
COORDINATES_DELIMITER = ',';

%% get file contents
fid = fopen(annotationFilePath, 'r');
contents = textscan(fid, '%s');
fclose(fid);

linesCell = contents{1};
objects = [];
cnt = 1;
for i = 1:length(linesCell)
   lineString = linesCell{i};
   
   %% parse line
   parsed = textscan(lineString, '%s', 'delimiter', DATA_DELIMITER);
   parsed = parsed{1};
   polygonString = parsed{1};
   charIDString = parsed{2};
   charStateString = parsed{3};
   
   if strcmp(charIDString, charID) ~= 1
       continue;
   end
   
   %% parse coordinates
   done = 0;
   xcoords = [];
   ycoords = [];
   remainString = polygonString;
   while ~done
       [xToken, remainString] = strtok(remainString, COORDINATES_DELIMITER);
       if isempty(strfind(remainString, POINTS_DELIMITER))
           [yToken, remainString] = strtok(remainString, DATA_DELIMITER);
           done = 1;
       else
           [yToken, remainString] = strtok(remainString, POINTS_DELIMITER);
       end
       
       xcoords = [xcoords; str2num(xToken)];
       ycoords = [ycoords; str2num(yToken)];
   end
   
   %% form object containing coordinates
   object.xcoords = xcoords;
   object.ycoords = ycoords;
   object.charID = charIDString;
   object.charState = charStateString;
   
   %% add to objects list
   objects{cnt} = object;
   cnt = cnt + 1;
end

end