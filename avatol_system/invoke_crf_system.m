function invoke_crf_system( inputPath, outputPath, options )
%INVOKE_SYSTEM Use this function to invoke the entire character scoring
%system.
%
%   inputPath   : path/.../sorted_input_data_<charID>_<charName>.txt
%   outputPath  : path/.../sorted_output_data_<charID>_<charName>.txt

%% ========== argument checking
narginchk(2, 3);

%% ========== default optional arguments
if nargin < 3
    options = struct;
end
% path to dataset directory, which is the root of media/ and annotations/
if ~isfield(options, 'DATASET_PATH')
    [basePath, ~, ~] = fileparts(inputPath);
    options.DATASET_PATH = basePath;
end
% path to write temporary files during processing
if ~isfield(options, 'TEMP_PATH')
    options.TEMP_PATH = 'temp-ignore';
end
% path to write log file for debugging and diagnostics
if ~isfield(options, 'LOG_FILE')
    options.LOG_FILE = [options.TEMP_PATH filesep 'log.txt'];
end
% path to the preprocessed data directory computed after
% the preprocessing step and used for HC-Search
if ~isfield(options, 'PREPROCESSED_PATH')
    options.PREPROCESSED_PATH = [options.TEMP_PATH filesep 'predata'];
end
% path to the results data directory after HC-Search runs
if ~isfield(options, 'HC_DETECTION_RESULTS_PATH')
    options.HC_DETECTION_RESULTS_PATH = [options.TEMP_PATH filesep 'results'];
end
% time bound parameter for HC-Search
if ~isfield(options, 'HCSEARCH_TIMEBOUND')
    options.HCSEARCH_TIMEBOUND = 1; % default results in IID classifier
end

%% ========== begin
if ~exist(options.TEMP_PATH, 'dir')
    mkdir(options.TEMP_PATH);
end
log_fid = fopen(options.LOG_FILE, 'a');

tglobalstart = tic;
writelog(log_fid, '==========\n');
writelog(log_fid, sprintf('Begin AVATOL system at %s.\n\n', datestr(now)));

%% ========== parse arguments of input file
writelog(log_fid, 'Parsing input file...\n');

% get character ID from file name
[~, inFileName, ~] = fileparts(inputPath);
[charID, charName] = get_char_id_from_file_name(inFileName);
charStateNames = containers.Map('KeyType', 'char', 'ValueType', 'any');

% get list of training and test instances
[trainingList, scoringList] = parse_input_file(inputPath);
scoringRange = 1+length(trainingList):length(trainingList)+length(scoringList);

telapsed = toc(tglobalstart);
writelog(log_fid, sprintf('Finished parsing input file. (%.1fs)\n\n', telapsed));

%% ========== preprocess data for HC-Search
tstart = tic;
writelog(log_fid, 'Preprocessing input data...\n');

% extract features, preprocess into data for HC-Search
color2label = containers.Map({0, 255}, {-1, 1});
[allData, charStateNames] = preprocess_avatol(options.DATASET_PATH, trainingList, scoringList, ...
    charID, charStateNames, color2label, options.PREPROCESSED_PATH);

telapsed = toc(tstart);
writelog(log_fid, sprintf('Finished preprocessing input data. (%.1fs)\n\n', telapsed));

%% ========== call HC-Search
tstart = tic;
writelog(log_fid, 'Running character detection...\n');

cmdlineArgs = sprintf('%s %s %d --learn --infer --prune none --ranker vw --successor flipbit-neighbors', ...
    options.PREPROCESSED_PATH, options.HC_DETECTION_RESULTS_PATH, options.HCSEARCH_TIMEBOUND);
if ispc
    fprintf('Detected PC. Running HC-Search...\n');
    [status, result] = dos(['hcsearch ' cmdlineArgs]);
else
    fprintf('Detected Unix. Running HC-Search...\n');
    [status, result] = unix(['./hcsearch ' cmdlineArgs]);
end
fprintf('status=\n\n%d\n\n', status);
fprintf('result=\n\n%s\n\n', result);

telapsed = toc(tstart);
writelog(log_fid, sprintf('Finished running character detection. (%.1fs)\n\n', telapsed));

%% ========== postprocess data for character scoring
tstart = tic;
writelog(log_fid, 'Running detection post-process...\n');

% postprocess
allData = postprocess_avatol(allData, sprintf('%s/results', options.HC_DETECTION_RESULTS_PATH), ...
    options.HCSEARCH_TIMEBOUND);

telapsed = toc(tstart);
writelog(log_fid, sprintf('Finished running detection post-process. (%.1fs)\n\n', telapsed));

%% ========== character scoring
tstart = tic;
writelog(log_fid, 'Running character scoring...\n');

cnt = 1;
for i = scoringRange
    % perform character scoring
    charState = score_basal_texture(allData{i});
    scoringList{cnt}.charState = charState;
    
    % save detection polygon
    [~, temp, ~] = fileparts(scoringList{cnt}.pathToMedia);
    pathToDetection = sprintf('%s/detection_results/%s_%s.txt', options.DATASET_PATH, temp, charID);
    convert_detection_to_annotation(pathToDetection, allData{i}, charID, charName, charState, charStateNames(charState));
    
    % save scores
    scoringList{cnt}.pathToDetection = pathToDetection;
    
    cnt = cnt + 1;
end

telapsed = toc(tstart);
writelog(log_fid, sprintf('Finished running character scoring. (%.1fs)\n\n', telapsed));

%% ========== save scores
tstart = tic;
writelog(log_fid, 'Saving scores...\n');

% save scores
write_scores(outputPath, trainingList, scoringList, {});

telapsed = toc(tstart);
writelog(log_fid, sprintf('Finished saving scores. (%.1fs)\n', telapsed));

%% ========== done
telapsed = toc(tglobalstart);
writelog(log_fid, sprintf('\nFinished AVATOL system at %s. Total elapsed time: %.1fs\n', datestr(now), telapsed));
writelog(log_fid, '==========\n\n');

fclose(log_fid);

end

function [charID, charName] = get_char_id_from_file_name(inFileName)

FILE_PREFIX = 'sorted_input_data_';

charID = inFileName(length(FILE_PREFIX)+1:end);
charID(strfind(charID, '_'):end) = [];

charName = inFileName(length([FILE_PREFIX charID '_'])+1:end);

end