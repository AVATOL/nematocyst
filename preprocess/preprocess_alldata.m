function preprocess_alldata( allData, outputPath, trainRange, validRange, testRange )
%PREPROCESS_ALLDATA Preprocesses allData cell struct variable into a
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
%   outputPath:	folder path to output preprocessed data
%                       e.g. 'DataPreprocessed/SomeDataset'
%   trainRange:	set range of training data
%   validRange:	set range of hold-out validation data
%   testRange:	set range of test data

%% argument checking
narginchk(5, 5);

%% parameters
NUM_CODE_WORDS = 100;

%% constants
TRAIN_LIST = 'Train.txt';
VALID_LIST = 'Validation.txt';
TEST_LIST = 'Test.txt';

%% create output folder
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end
if ~exist([outputPath '/nodes/'], 'dir')
    mkdir([outputPath '/nodes/']);
end
if ~exist([outputPath '/edges/'], 'dir')
    mkdir([outputPath '/edges/']);
end
if ~exist([outputPath '/segments/'], 'dir')
    mkdir([outputPath '/segments/']);
end
if ~exist([outputPath '/meta/'], 'dir')
    mkdir([outputPath '/meta/']);
end
if ~exist([outputPath '/splits/'], 'dir')
    mkdir([outputPath '/splits/']);
end

%% initialize
trainListFile = [outputPath '/splits/' TRAIN_LIST];
validListFile = [outputPath '/splits/' VALID_LIST];
testListFile = [outputPath '/splits/' TEST_LIST];

train_fid = fopen(trainListFile, 'w');
valid_fid = fopen(validListFile, 'w');
test_fid = fopen(testListFile, 'w');

nFiles = length(allData);
classes = [];

bowData = [];

%% write files
for i = 1:nFiles
    fprintf('Exporting example %d...\n', i-1);
    
    filename = sprintf('%d', i-1);
    if isfield(allData{i}, 'filename');
        filename = allData{i}.filename;
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
    
    nodesFile = sprintf('%s.txt', filename);
    edgesFile = sprintf('%s.txt', filename);
    segmentsFile = sprintf('%s.txt', filename);
    metaFile = sprintf('%s.txt', filename);
    
    % write nodes
    libsvmwrite([outputPath '/nodes/' nodesFile], allData{i}.segLabels, sparse(allData{i}.feat2));
    
    % write edges
    [ai,aj,aval] = find(allData{i}.adj);
    spAdj = [ai,aj,aval];
    dlmwrite([outputPath '/edges/' edgesFile], spAdj, ' ');
    
    % write segments
    dlmwrite([outputPath '/segments/' segmentsFile], allData{i}.segs2, ' ');
    
    classes = union(classes, allData{i}.segLabels);
    
    %% write meta file
    fid = fopen([outputPath '/meta/' metaFile], 'w');
    fprintf(fid, 'nodes=%d\n', size(allData{i}.feat2, 1));
    fprintf(fid, 'features=%d', size(allData{i}.feat2, 2));
    fprintf(fid, 'height=%d\n', size(allData{i}.segs2, 1));
    fprintf(fid, 'width=%d', size(allData{i}.segs2, 2));
    fclose(fid);
    
    %% append to bow data
    bowData = horzcat(bowData, allData{i}.feat2');
end

fclose(train_fid);
fclose(valid_fid);
fclose(test_fid);

%% write metadata file
fid = fopen([outputPath '/metadata.txt'], 'w');
fprintf(fid, 'num=%d\n', nFiles);
writeRange(fid, 'classes', classes);
writeRange(fid, 'backgroundclasses', []);
fclose(fid);

%% save copy of matlab variables
fprintf('Saving variable allData to file...\n');
save([outputPath '/allData.mat'], 'allData');

%% create initial classifier training file
INITFUNC_TRAINING_FILE = 'initfunc_training.txt';
INITFUNC_TEMP_FILE = 'initfunc_temp.txt';
if ispc
    outputPathWin = strrep(outputPath, '/', '\');
    dos(['copy NUL ' outputPathWin '\' INITFUNC_TRAINING_FILE]);
elseif isunix
    outputPathLinux = strrep(outputPath, '\', '/');
    unix(['touch ' outputPathLinux '/' INITFUNC_TRAINING_FILE]);
end
for i = trainRange
    file = sprintf('%d.txt', i-1);
    if isfield(allData{i}, 'filename');
        file = sprintf('%s.txt', allData{i}.filename);
    end
    
    if ispc
       outputPathWin = strrep(outputPath, '/', '\');
       typeCmd = ['type ' outputPathWin '\' INITFUNC_TRAINING_FILE ' '...
           outputPathWin '\nodes\' file ' > ' outputPathWin '\' INITFUNC_TEMP_FILE];
       delCmd = ['del ' outputPathWin '\' INITFUNC_TRAINING_FILE];
       renameCmd = ['move ' outputPathWin '\' INITFUNC_TEMP_FILE ' '...
           outputPathWin '\' INITFUNC_TRAINING_FILE];
       dos(typeCmd);
       dos(delCmd);
       dos(renameCmd);
    elseif isunix
       outputPathLinux = strrep(outputPath, '\', '/');
       typeCmd = ['cat ' outputPathLinux '/' INITFUNC_TRAINING_FILE ' '...
           outputPathLinux '\nodes\' file ' > ' outputPathLinux '/' INITFUNC_TEMP_FILE];
       delCmd = ['rm -f ' outputPathLinux '/' INITFUNC_TRAINING_FILE];
       renameCmd = ['mv ' outputPathLinux '/' INITFUNC_TEMP_FILE ' '...
           outputPathLinux '/' INITFUNC_TRAINING_FILE];
       unix(typeCmd);
       unix(delCmd);
       unix(renameCmd);
    end
end

%% generate codebook and write to file
fprintf('generating codebook...\n');
[centers, ~] = vl_kmeans(bowData, NUM_CODE_WORDS);
dlmwrite([outputPath '/codebook.txt'], centers');

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