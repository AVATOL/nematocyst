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