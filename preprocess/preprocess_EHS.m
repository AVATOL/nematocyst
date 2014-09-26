function [ allData ] = preprocess_EHS( allData, EHSPath, outputPath, trainRange, validRange, testRange )
%PREPROCESS_EHS Preprocesses allData cell struct variable into a
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
%   validRange:	set range of hold-out validation data
%   testRange:	set range of test data

%% argument checking
narginchk(6, 6);

outputPath = normalize_file_sep(outputPath);
EHSPath = normalize_file_sep(EHSPath);

%% parameters
NUM_CODE_WORDS = 100;

%% constants
TRAIN_LIST = 'Train.txt';
VALID_LIST = 'Validation.txt';
TEST_LIST = 'Test.txt';
ALL_LIST = 'All.txt';

% example: 'DataRaw/SB_features/iccv09_%d_node_edge.mat'
NODE_EDGE_STRING = EHSPath;
if strcmp(filesep, '\\') == 0
    NODE_EDGE_STRING = strrep(NODE_EDGE_STRING, '\', '\\');
end

EXTERNAL_PATH = 'external';
LIBLINEAR_PATH = [EXTERNAL_PATH filesep 'liblinear'];

%% create output folder
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end
if ~exist([outputPath filesep 'nodes' filesep], 'dir')
    mkdir([outputPath filesep 'nodes' filesep]);
end
if ~exist([outputPath filesep 'nodelocations' filesep], 'dir')
    mkdir([outputPath filesep 'nodelocations' filesep]);
end
if ~exist([outputPath filesep 'edges' filesep], 'dir')
    mkdir([outputPath filesep 'edges' filesep]);
end
if ~exist([outputPath filesep 'edgefeatures' filesep], 'dir')
    mkdir([outputPath filesep 'edgefeatures' filesep]);
end
if ~exist([outputPath filesep 'segments' filesep], 'dir')
    mkdir([outputPath filesep 'segments' filesep]);
end
if ~exist([outputPath filesep 'groundtruth' filesep], 'dir')
    mkdir([outputPath filesep 'groundtruth' filesep]);
end
if ~exist([outputPath filesep 'meta' filesep], 'dir')
    mkdir([outputPath filesep 'meta' filesep]);
end
if ~exist([outputPath filesep 'splits' filesep], 'dir')
    mkdir([outputPath filesep 'splits' filesep]);
end
if ~exist([outputPath filesep 'initstate' filesep], 'dir')
    mkdir([outputPath filesep 'initstate' filesep]);
end

%% initialize
trainListFile = [outputPath filesep 'splits' filesep TRAIN_LIST];
validListFile = [outputPath filesep 'splits' filesep VALID_LIST];
testListFile = [outputPath filesep 'splits' filesep TEST_LIST];
allListFile = [outputPath filesep 'splits' filesep ALL_LIST];

train_fid = fopen(trainListFile, 'w');
valid_fid = fopen(validListFile, 'w');
test_fid = fopen(testListFile, 'w');
all_fid = fopen(allListFile, 'w');

nFiles = length(allData);
classes = [];

bowData = [];

%% write files
trainNodeLabels = [];
trainNodeFeatures = [];
trainEdgeLabels = [];
trainEdgeFeatures = [];
for i = 1:nFiles
    fprintf('Exporting example %d...\n', i-1);
    
    %% load E, H, S
    load(sprintf(NODE_EDGE_STRING, i));
    E_symmetric = E + E';
    
    filename = sprintf('%d', i-1);
    if isfield(allData{i}, 'filename');
        filename = allData{i}.filename;
    else
        allData{i}.filename = filename;
    end
    
    if ismember(i, trainRange)
        fprintf(train_fid, '%s\n', filename);
    elseif ismember(i, validRange)
        fprintf(valid_fid, '%s\n', filename);
    elseif ismember(i, testRange)
        fprintf(test_fid, '%s\n', filename);
    else
        % error!
    end
    fprintf(all_fid, '%s\n', filename);
    
    nodesFile = sprintf('%s.txt', filename);
    nodeLocationsFile = sprintf('%s.txt', filename);
    edgesFile = sprintf('%s.txt', filename);
    edgeFeaturesFile = sprintf('%s.txt', filename);
    segmentsFile = sprintf('%s.txt', filename);
    groundtruthFile = sprintf('%s.txt', filename);
    metaFile = sprintf('%s.txt', filename);
    
    %% read
    nodeFeat = allData{i}.feat2;
    nodeLabel = allData{i}.segLabels;
    segments = allData{i}.segs2;
    
    % write nodes
    libsvmwrite([outputPath filesep 'nodes' filesep nodesFile], nodeLabel, sparse(nodeFeat));
    trainNodeLabels = [trainNodeLabels; nodeLabel];
    trainNodeFeatures = [trainNodeFeatures; nodeFeat];
    
    % write node locations and sizes
    if isfield(allData{i}, 'segLocations') && isfield(allData{i}, 'segSizes');
        nodeLocations = allData{i}.segLocations;
        nodeSizes = allData{i}.segSizes;
    else
        [nodeLocations, nodeSizes] = pre_extract_node_locations(segments, length(nodeLabel));
        allData{i}.segLocations = nodeLocations;
        allData{i}.segSizes = nodeSizes;
    end
    
    dlmwrite([outputPath filesep 'nodelocations' filesep nodeLocationsFile], [nodeLocations nodeSizes], ' ');
    
    % write edges
    [ai,aj,aval] = find(E_symmetric);
    spAdj = [ai,aj,aval];
    dlmwrite([outputPath filesep 'edges' filesep edgesFile], spAdj, ' ');
    
    % write edge features
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
    
    libsvmwrite([outputPath filesep 'edgefeatures' filesep edgeFeaturesFile], edgeLabels, sparse(edgeFeatures));
    trainEdgeLabels = [trainEdgeLabels; edgeLabels];
    trainEdgeFeatures = [trainEdgeFeatures; edgeFeatures];
    
    % write segments
    dlmwrite([outputPath filesep 'segments' filesep segmentsFile], segments, ' ');
    
    % write ground truth
    dlmwrite([outputPath filesep 'groundtruth' filesep groundtruthFile], allData{i}.labels, ' ');
    
    classes = union(classes, nodeLabel);
    
    %% write meta file
    fid = fopen([outputPath filesep 'meta' filesep metaFile], 'w');
    fprintf(fid, 'nodes=%d\n', size(nodeFeat, 1));
    fprintf(fid, 'features=%d\n', size(nodeFeat, 2));
    fprintf(fid, 'height=%d\n', size(segments, 1));
    fprintf(fid, 'width=%d', size(segments, 2));
    fclose(fid);
    
    %% append to bow data
    bowData = horzcat(bowData, nodeFeat');
end

fclose(train_fid);
fclose(valid_fid);
fclose(test_fid);
fclose(all_fid);

%% write metadata file
fid = fopen([outputPath filesep 'metadata.txt'], 'w');
fprintf(fid, 'num=%d\n', nFiles);
writeRange(fid, 'classes', classes);
writeRange(fid, 'backgroundclasses', []);
writeRange(fid, 'ignoreclasses', []);
fclose(fid);

%% create initial classifier training file
INITFUNC_TRAINING_FILE = 'initfunc_training.txt';
libsvmwrite([outputPath filesep INITFUNC_TRAINING_FILE], trainNodeLabels, sparse(trainNodeFeatures));

%% create edge classifier training file
EDGECLASSIFIER_TRAINING_FILE = 'edgeclassifier_training.txt';
libsvmwrite([outputPath filesep EDGECLASSIFIER_TRAINING_FILE], trainEdgeLabels, sparse(trainEdgeFeatures));

%% train initial prediction classifier on the training file just generated
INITFUNC_MODEL_FILE = 'initfunc_model.txt';
fprintf('Training initial classifier model...\n');
if ispc
    LIBLINEAR_TRAIN = [LIBLINEAR_PATH filesep 'windows' filesep 'train'];
elseif isunix
    LIBLINEAR_TRAIN = [LIBLINEAR_PATH filesep 'train'];
end

LIBLINEAR_TRAIN_CMD = [LIBLINEAR_TRAIN ' -s 7 -c 10 ' ...
    outputPath filesep INITFUNC_TRAINING_FILE ' ' ...
    outputPath filesep INITFUNC_MODEL_FILE];

if ispc
    dos(LIBLINEAR_TRAIN_CMD);
elseif isunix
    unix(LIBLINEAR_TRAIN_CMD);
end
initStateModel = train(trainNodeLabels, sparse(trainNodeFeatures), '-s 7 -c 10');

%% generate the initial prediction files
for i = 1:nFiles
    fprintf('Predicting example %d...\n', i-1);
    
    filename = sprintf('%d', i-1);
    if isfield(allData{i}, 'filename');
        filename = allData{i}.filename;
    else
        allData{i}.filename = filename;
    end
    
    initPredFile = sprintf('%s.txt', filename);
    nodesFile = sprintf('%s.txt', filename);
    
    if ispc
        LIBLINEAR_PREDICT = [LIBLINEAR_PATH filesep 'windows' filesep 'predict'];
    elseif isunix
        LIBLINEAR_PREDICT = [LIBLINEAR_PATH filesep 'predict'];
    end
    
    LIBLINEAR_PREDICT_CMD = [LIBLINEAR_PREDICT ' -b 1 ' ...
        outputPath filesep 'nodes' filesep nodesFile ' ' ...
        outputPath filesep INITFUNC_MODEL_FILE ' ' ...
        outputPath filesep 'initstate' filesep initPredFile];
    
    if ispc
        dos(LIBLINEAR_PREDICT_CMD);
    elseif isunix
        unix(LIBLINEAR_PREDICT_CMD);
    end
    
    [initStateLabels, ~, ~] = predict(allData{i}.segLabels, sparse(allData{i}.feat2), initStateModel, '-b 1');
    allData{i}.initState = initStateLabels;
end

%% generate codebook and write to file
fprintf('generating codebook...\n');
[centers, ~] = vl_kmeans(bowData, NUM_CODE_WORDS);
dlmwrite([outputPath filesep 'codebook.txt'], centers');

%% save copy of matlab variables
fprintf('Saving variable allData to file...\n');
save([outputPath filesep 'allData.mat'], 'allData', '-v7.3');

%% train initial prediction classifier on the edge training file just generated
EDGECLASSIFIER_MODEL_FILE = 'edgeclassifier_model.txt';
fprintf('Training initial classifier model...\n');
if ispc
    LIBLINEAR_TRAIN = [LIBLINEAR_PATH filesep 'windows' filesep 'train'];
elseif isunix
    LIBLINEAR_TRAIN = [LIBLINEAR_PATH filesep 'train'];
end

LIBLINEAR_TRAIN_CMD = [LIBLINEAR_TRAIN ' -s 7 -c 10 ' ...
    outputPath filesep EDGECLASSIFIER_TRAINING_FILE ' ' ...
    outputPath filesep EDGECLASSIFIER_MODEL_FILE];

if ispc
    dos(LIBLINEAR_TRAIN_CMD);
elseif isunix
    unix(LIBLINEAR_TRAIN_CMD);
end

end

function writeRange(fid, setType, range)

fprintf(fid, '%s=', setType);
for i = 1:length(range)
    val = range(i);
    fprintf(fid, '%d', val);
    if val ~= range(end)
        fprintf(fid, ',');
    end
end
fprintf(fid, '\n');

end