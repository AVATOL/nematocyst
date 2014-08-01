function [ allData ] = postprocess_avatol( allData, resultsDir, timeBound )
%POSTPROCESS_RESULTS Read results from HC-Search into allData.
%
%	allData:        data structure containing all preprocessed data
%   resultsDir:     results folder containing HC-Search results
%   timeBound:      time bound from HC-Search

%% read from HC-Search results
for i = 1:length(allData)
    allDataInstance = allData{i};
    if isfield(allDataInstance, 'segLabels')
        continue;
    end
    
    %% read from file
    segLabels = dlmread([resultsDir filesep sprintf('final_nodes_hc_test_time%d_fold%d_%s.txt', timeBound, 0, allDataInstance.filename)]);
    
    %% assign to allData structure
    allData{i}.inferLabels = segLabels;
    allData{i}.inferImg = infer_pixels(segLabels, allDataInstance.segs2);
end

end

function [inferPixels] = infer_pixels(inferLabels, segments)

inferPixels = zeros(size(segments));
nNodes = length(inferLabels);

for i = 1:nNodes
    temp = segments;
    temp(temp ~= i) = 0;
    temp(temp == i) = inferLabels(i);
    
    inferPixels = inferPixels + temp;
end

end