function convert_detection_to_annotation( pathToDetection, allDataInstance, charID, charName, charState, charStateName )
%CONVERT_DETECTION_TO_ANNOTATION Summary of this function goes here
%   Detailed explanation goes here

BACKGROUND_LABEL = -1;
FOREGROUND_LABEL = 1;

labels = allDataInstance.inferImg;
labels(labels == BACKGROUND_LABEL) = 0;
labels(labels == FOREGROUND_LABEL) = 1;
objList = bwboundaries(labels);

objects = cell(length(objList), 1);
for i = 1:length(objList)
    obj = objList{i};
    
    xcoord = obj(:, 2);
    ycoord = obj(:, 1);
    
    objects{i}.xcoords = floor(100.0 * xcoord / size(labels, 2));
    objects{i}.ycoords = floor(100.0 * ycoord / size(labels, 1));
    objects{i}.charID = charID;
    objects{i}.charName = charName;
    objects{i}.charState = num2str(charState);
    objects{i}.charStateName = charStateName;
end

save_annotation_polygon(objects, pathToDetection);

end
