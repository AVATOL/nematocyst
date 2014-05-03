function generate_edge_weights( allData, outputPath, trainRange )
%GENERATE_EDGE_WEIGHTS Preprocesses allData cell struct variable into a
%data format for HCSearch to work. Features are already extracted in
%allData.
%
%   allData:	data structure containing all preprocessing data
%                   allData{i}.img mxnx3 uint8
%                   allData{i}.labels mxn double
%                   allData{i}.segs2 mxn double
%                   allData{i}.feat2 sxd double
%                   allData{i}.segLabels sx1 double
%                   allData{i}.adj sxs logical
%                   allData{i}.filename string (optional)
%                   allData{i}.segLocations sx2 double (optional)
%   outputPath:	folder path to output preprocessed data
%                       e.g. 'DataPreprocessed/SomeDataset'
%   trainRange:	set range of training data

%% argument checking
narginchk(5, 5);

%% create output folder
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end
if ~exist([outputPath '/edgesweights/'], 'dir')
    mkdir([outputPath '/edgesweights/']);
end

fprintf('Training edge model...\n');
features = [];
labels = [];

for i = trainRange
    fprintf('\timage %d\n', i);

    [ai,aj,~] = find(allData{i}.adj);

    featDim = size(allData{i}.feat2, 2);
    edgeFeatures = zeros(length(ai), featDim);
    edgeLabels = zeros(length(ai), 1);
    for j = 1:length(ai)
        e1 = allData{i}.feat2(ai(j), :);
        e2 = allData{i}.feat2(aj(j), :);
        edgeFeatures(j, :) = abs(e1 - e2);
        edgeLabels(j, 1) = allData{i}.segLabels(ai(j), 1) == allData{i}.segLabels(aj(j), 1);
    end
    edgeLabels(edgeLabels == 0) = -1;

    features = [features; edgeFeatures];
    labels = [labels; edgeLabels];
end
features = sparse(features);
fprintf('Training model...\n');
model = train(labels, features, '-s 7 -c 10');

%% write files
nFiles = length(allData);
for i = 1:nFiles
    fprintf('Exporting example %d...\n', i-1);
    
    filename = sprintf('%d', i-1);
    if isfield(allData{i}, 'filename');
        filename = allData{i}.filename;
    end
    edgesFile = sprintf('%s.txt', filename);
    
    [ai,aj,~] = find(allData{i}.adj);
    
    featDim = size(allData{i}.feat2, 2);
    edgeFeatures = zeros(length(ai), featDim);
    edgeLabels = zeros(length(ai), 1);
    for j = 1:length(ai)
        e1 = allData{i}.feat2(ai(j), :);
        e2 = allData{i}.feat2(aj(j), :);
        edgeFeatures(j, :) = abs(e1 - e2);
        edgeLabels(j, 1) = allData{i}.segLabels(ai(j), 1) == allData{i}.segLabels(aj(j), 1);
    end
    edgeLabels(edgeLabels == 0) = -1;
    
    %% get probabilities
    features = edgeFeatures;
    gtLabels = edgeLabels;
    [~, ~, probs] = predict(gtLabels, sparse(features), model, '-b 1');
    
    spAdj = [ai aj probs(:, 1)];
    dlmwrite([outputPath 'edgeweights/' edgesFile], spAdj, ' ');
end

end
