function write_scores( outputPath, trainingList, scoringList, nonScoringList )
%WRITE_SCORES Write scores to output file.
%
%   outputPath      :   path/.../sorted_output_data_<charID>_<charName>.txt
%   trainingList    :   cell of structs where a struct denotes 
%                       a training instance and each struct has
%                           .pathToMedia: path to image
%                           .charState: character state
%                           .pathToAnnotation: path to annotation
%   scoringList     :   cell of structs where a struct denotes 
%                       a test/scoring instance and each struct has
%                           .pathToMedia: path to image
%                           .charState: character state inferred
%                           .pathToDetection: path to detection
%   nonScoringList  :   cell of structs where a struct denotes 
%                       a non-scored test instance and each struct has
%                           .pathToMedia: path to image

fid = fopen(outputPath, 'w');

for i = 1:length(trainingList)
    data = trainingList{i};
    fprintf(fid, 'training_data:%s:%s:%s\n', data.pathToMedia, data.charState, data.pathToAnnotation);
end

for i = 1:length(scoringList)
    data = scoringList{i};
    fprintf(fid, 'image_scored:%s:%s:%s\n', data.pathToMedia, data.charState, data.pathToDetection);
end

for i = 1:length(nonScoringList)
    data = nonScoringList{i};
    fprintf(fid, 'image_not_scored:%s\n', data.pathToMedia);
end

fclose(fid);

end

