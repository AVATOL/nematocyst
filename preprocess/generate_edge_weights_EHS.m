function generate_edge_weights_EHS( allData, EHSPath, outputPath, trainRange )
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
%   EHSPath:    folder path to edge potentials (.mat files contain E, H, S)
%   outputPath:	folder path to output preprocessed data
%                       e.g. 'DataPreprocessed/SomeDataset'
%   trainRange:	set range of training data

%% argument checking
narginchk(4, 4);

EHSPath = normalize_file_sep(EHSPath);

% example: 'DataRaw/SB_features/iccv09_%d_node_edge.mat'
NODE_EDGE_STRING = EHSPath;
if strcmp(filesep, '\\') == 0
    NODE_EDGE_STRING = strrep(NODE_EDGE_STRING, '\', '\\');
end

%% create output folder
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end
if ~exist([outputPath '/edgeweights/'], 'dir')
    mkdir([outputPath '/edgeweights/']);
end

fprintf('Training edge model...\n');
features = [];
labels = [];

for i = trainRange
    fprintf('\timage %d\n', i);

    %% load E, H, S
    load(sprintf(NODE_EDGE_STRING, i));
    E_symmetric = E + E';
    
    [ai,aj,~] = find(E_symmetric);
    
    %% read
    nodeLabel = allData{i}.segLabels;

    featDim = 4;
    edgeFeatures = zeros(length(ai), featDim);
    edgeLabels = zeros(length(ai), 1);
    for j = 1:length(ai)
        if E(ai(j), aj(j)) == 0
            e1 = aj(j);
            e2 = ai(j);
        else
            e1 = ai(j);
            e2 = aj(j);
        end
        
        edgeFeatures(j, :) = S{e1, e2};
        edgeLabels(j, 1) = nodeLabel(e1, 1) ~= nodeLabel(e2, 1);
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
    
    %% load E, H, S
    load(sprintf(NODE_EDGE_STRING, i));
    E_symmetric = E + E';
    
    [ai,aj,~] = find(E_symmetric);
    
    %% read
    nodeLabel = allData{i}.segLabels;
    
    featDim = 4;
    edgeFeatures = zeros(length(ai), featDim);
    edgeLabels = zeros(length(ai), 1);
    for j = 1:length(ai)
        if E(ai(j), aj(j)) == 0
            e1 = aj(j);
            e2 = ai(j);
        else
            e1 = ai(j);
            e2 = aj(j);
        end
        
        edgeFeatures(j, :) = S{e1, e2};
        edgeLabels(j, 1) = nodeLabel(e1, 1) ~= nodeLabel(e2, 1);
    end
    edgeLabels(edgeLabels == 0) = -1;
    
    %% get probabilities
    features = edgeFeatures;
    gtLabels = edgeLabels;
    [~, ~, probs] = predict(gtLabels, sparse(features), model, '-b 1');
    
    spAdj = [ai aj probs(:, 1)];
    dlmwrite([outputPath '/edgeweights/' edgesFile], spAdj, ' ');
end

end
