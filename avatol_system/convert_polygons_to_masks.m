function convert_polygons_to_masks( basePath, trainingList, charID, outputPath )
%CONVERT_POLYGONS_TO_MASKS 
%
%   trainingList    :   cell of structs where a struct denotes 
%                       a training instance and each struct has
%                           .pathToMedia: path to image
%                           .charState: character state
%                           .pathToAnnotation: path to annotation
%                           .taxonID: taxon ID
%   charID          :   character ID string
%   outputPath      :   path to save masks

%% argument checking
narginchk(4, 4);

%% constants
MASK_EXTENSION = '.png';

%% create output directory if doesn't exist
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end

%% process images folder
for i = 1:length(trainingList)
    imagesPath = [basePath filesep trainingList{i}.pathToMedia];
    annotationPath = [basePath filesep trainingList{i}.pathToAnnotation];
    [~, annotationFileName, ~] = fileparts(annotationPath);
    
    %% read polygon coordinates data
    fprintf('Processing mask %d...\n', i);
    objects = read_annotation_file(annotationPath, charID);
    
    %% convert to mask
    img = imread(imagesPath);
    imgMask = polygons2masks(img, objects);
    
    %% save mask to file
    imwrite(imgMask, [outputPath filesep annotationFileName MASK_EXTENSION]);
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