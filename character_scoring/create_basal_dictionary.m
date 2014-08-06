function [ dictionary ] = create_basal_dictionary( allData, numCodeWords )
%CREATE_BASAL_DICTIONARY Get k-means dictionary of basal features

BASAL_LABEL = 1;

allFeatures = [];

for i = 1:length(allData)
   allDataInstance = allData{i};
   features = allDataInstance.feat2;
   labels = allDataInstance.segLabels;
   
   labels(labels == BASAL_LABEL) = 1;
   labels(labels ~= BASAL_LABEL) = 0;
   labels = ~labels;
   
   [rows, cols] = find(labels);
   
   features(rows, :) = [];
   
   allFeatures = [allFeatures features'];
end

[dictionary, ~] = vl_kmeans(allFeatures, numCodeWords);

end

