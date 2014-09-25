function [ trainingList, scoringList ] = parse_input_file( inputPath )
%PARSE_INPUT_FILE Parse the input file into data structures.
%
%   inputPath       :   path/.../sorted_input_data_<charID>_<charName>.txt
%   trainingList    :   cell of structs where a struct denotes 
%                       a training instance and each struct has
%                           .pathToMedia: path to image
%                           .charState: character state
%                           .pathToAnnotation: path to annotation
%                           .taxonID: taxon ID
%   scoringList     :   cell of structs where a struct denotes 
%                       a test/scoring instance and each struct has
%                           .pathToMedia: path to image
%                           .taxonID: taxon ID

%% argument checking
narginchk(1, 1);

%% constants
DATA_DELIMITER = ':';
TRAINING_DATA_STRING = 'training_data';
IMAGE_TO_SCORE_STRING = 'image_to_score';

%% initialization
trainingList = {};
scoringList = {};

%% get contents of file
if ~exist(inputPath, 'file')
    error('Input file "%s" does not exist!', inputPath);
end

fid = fopen(inputPath, 'r');
contents = textscan(fid, '%s');
fclose(fid);
linesCell = contents{1};

trainCnt = 1;
scoreCnt = 1;
for i = 1:length(linesCell)
   lineString = linesCell{i};
   
   parsed = textscan(lineString, '%s', 'delimiter', DATA_DELIMITER);
   parsed = parsed{1};
   
   object = struct;
   if strcmp(parsed{1}, TRAINING_DATA_STRING) == 1
       object.pathToMedia = fullfile(parsed{2});
       object.charState = parsed{3};
       object.charStateName = parsed{4};
       object.pathToAnnotation = fullfile(parsed{5});
       object.taxonID = parsed{6};
       object.lineNumber = parsed{7};
       
       trainingList{trainCnt} = object;
       trainCnt = trainCnt+1;
       
   elseif strcmp(parsed{1}, IMAGE_TO_SCORE_STRING) == 1
       object.pathToMedia = fullfile(parsed{2});
       object.taxonID = parsed{3};
       
       scoringList{scoreCnt} = object;
       scoreCnt = scoreCnt+1;
       
   else
      error('Invalid token.'); 
   end
end

end

