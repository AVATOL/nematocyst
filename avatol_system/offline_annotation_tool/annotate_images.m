function annotate_images( imagesPath, outputPath )
%ANNOTATE_IMAGES Lightweight polyon annotation tool for characters.
%
%   imagesPath:     path to images
%   outputPath:     path to save annotations

%% argument checking
narginchk(2, 2);

%% other settings
% valid image extensions to check - specify each beginning with .
VALID_IMAGE_EXTENSIONS = lower({'.jpg'; '.png'});

%% create output directory if doesn't exist
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end

%% begin annotation
ready = questdlg(['We will start annotating images. For each image a window will pop up accepting annotations. '...
    'To add a point to the polygon, click on the image to set a point. Continue until you are done. '...
    'When you are done, right click inside the region and click "Create Mask" to accept the annotation. '...
    'To reject the mask and try again, hit the [Escape] key. '...
    'After accepting the annotation, give it a character ID and state, then continue. '...
    'Are you ready?']);
if strcmp(ready, 'Yes') ~= 1
    fprintf('Exiting prematurely...\n');
    return; 
end

%% process images/groundtruth folder
fileListingArray = dir(imagesPath);
for i= 1:length(fileListingArray)
    fileStruct = fileListingArray(i);
    
    %% skip folders and files with invalid extensions
    if fileStruct.isdir
        continue;
    else
        [~,fileName,ext] = fileparts(fileStruct.name);
        if sum(ismember(VALID_IMAGE_EXTENSIONS, lower(ext))) == 0
            continue;
        end
    end
    
    %% begin user interactive annotation session
    imgPath = [imagesPath filesep fileStruct.name];
    img = imread(imgPath);
    objects = annotate_image_polygon(img);
    
    %% save annotation data to file
    save_annotation(objects, [outputPath filesep fileName '.txt']);
end

end

function save_annotation(objects, filename)
% saves annotation objects to file
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