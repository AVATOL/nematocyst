function [ allData ] = postprocess_avatol( allData, resultsDir, splitsPath, timeBound )
%POSTPROCESS_RESULTS Read results from HC-Search into allData.
%
%	allData:        data structure containing all preprocessed data
%   resultsDir:     results folder containing HC-Search results
%   splitsPath:     path to Test.txt splits file
%   timeBound:      time bound from HC-Search

%% read splits file
if ~exist(splitsPath, 'file')
    error('test splits file does not exist: %s', splitsPath);
end
fid = fopen(splitsPath, 'r');
list = textscan(fid, '%s');
fclose(fid);
testList = list{1,1};

%% read from HC-Search results
for i = 1:length(testList)
    index = str2num(testList{i});
    
    %% read from file
    segLabels = dlmread([resultsDir filesep sprintf('final_nodes_hc_test_time%d_fold%d_%d.txt', timeBound, 0, index)]);
    
    %% assign to allData structure
    allData{index + 1}.inferLabels = segLabels;
    allData{index + 1}.inferImg = infer_pixels(segLabels, allData{index + 1}.segs2);
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