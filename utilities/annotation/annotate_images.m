function annotate_images( imagesPath, splitsPath, outputPath )
%ANNOTATE_IMAGES Annotate characters using polygons
%   imagesPath:     path to images
%   splitsPath:     path to splits
%   outputPath:     path to save annotations

%% argument checking
narginchk(3, 3);

%% other settings
TRAIN_LIST = 'Train.txt';
VALID_LIST = 'Validation.txt';
TEST_LIST = 'Test.txt';
IMAGE_EXT = '.jpg';

%% process list of training, validation, test splits
trainListFile = [splitsPath filesep TRAIN_LIST];
validListFile = [splitsPath filesep VALID_LIST];
testListFile = [splitsPath filesep TEST_LIST];

if ~exist(trainListFile, 'file')
    error('training file does not exist: %s', trainListFile);
end
fid = fopen(trainListFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
trainList = list{1,1};
trainRange = 1:length(trainList);

if ~exist(validListFile, 'file')
    error('validation file does not exist: %s', validListFile);
end
fid = fopen(validListFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
validList = list{1,1};
validRange = length(trainList)+1:length(trainList)+length(validList);

if ~exist(testListFile, 'file')
    error('test file does not exist: %s', testListFile);
end
fid = fopen(testListFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
testList = list{1,1};
testRange = length(trainList)+length(validList)+1:length(trainList)+length(validList)+length(testList);

fileArray = [trainList; validList; testList];

if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end

%% process images/groundtruth folder
nImages = length(fileArray);
cnt = 1;
for i= 1:length(fileArray)
    file = fileArray{i, 1};
    
    %% get paths
    imgPath = [imagesPath filesep file IMAGE_EXT];
    fprintf('Processing image %s...\n', file);
    
    if ~exist(imgPath, 'file')
        error(['image "' imgPath '" does not exist']);
    end
    
    %% get image
    img = imread(imgPath);
    
    %% annotate polygons
    objects = annotate_image_polygon(img);
    
    %% save
    save_annotation(objects, [outputPath filesep file '.txt']);
    
    cnt = cnt+1;
end

end

function save_annotation(objects, filename)
% format: x1,y1;...;xn,yn:charID:charState

fid = fopen(filename, 'w');

for i = 1:length(objects)
   obj = objects{i};
   
   string = sprintf('%.f,%.f', obj.xcoords(1), obj.ycoords(1)); 
   for j = 2:length(obj.xcoords)
      string = strcat(string, sprintf(';%.f,%.f', obj.xcoords(j), obj.ycoords(j))); 
   end
   string = sprintf('%s:%s:%s', string, obj.charID, obj.charState);
   
   fprintf(fid, '%s\n', string);
end

fclose(fid);

end