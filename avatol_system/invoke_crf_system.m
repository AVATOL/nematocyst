function invoke_crf_system( inputPath, outputPath, options )
%INVOKE_SYSTEM Use this function to invoke the entire character scoring
%system.
%
%   inputPath   : path/.../sorted_input_data_<charID>_<charName>.txt
%   outputPath  : path/.../sorted_output_data_<charID>_<charName>.txt

%% ========== argument checking
narginchk(2, 3);

% argument cleanup
inputPath = fullfile(inputPath);
outputPath = fullfile(outputPath);

%% ========== default optional arguments
if nargin < 3
    options = struct;
end
% path to the base directory, which contains the HCSearch executable file
% if not '', then must end in filesep
if ~isfield(options, 'BASE_PATH')
    options.BASE_PATH = '';
%     options.BASE_PATH = ['..' filesep];
end
options.BASE_PATH = fullfile(options.BASE_PATH);
% path to dataset directory, which is the root of media/ and annotations/
if ~isfield(options, 'DATASET_PATH')
    [inputBasePath, ~, ~] = fileparts(inputPath);
    options.DATASET_PATH = fullfile(inputBasePath);
%     options.DATASET_PATH = pwd;
end
options.DATASET_PATH = fullfile(options.DATASET_PATH);
% path to write temporary files during processing
if ~isfield(options, 'TEMP_PATH')
    options.TEMP_PATH = 'temp-ignore';
end
options.TEMP_PATH = fullfile(options.TEMP_PATH);
% path to write log file for debugging and diagnostics
if ~isfield(options, 'LOG_FILE')
    options.LOG_FILE = fullfile(options.TEMP_PATH, 'log.txt');
end
options.LOG_FILE = fullfile(options.LOG_FILE);
% path to the preprocessed data directory computed after
% the preprocessing step and used for HC-Search
if ~isfield(options, 'PREPROCESSED_PATH')
    options.PREPROCESSED_PATH = fullfile(options.TEMP_PATH, 'predata');
end
options.PREPROCESSED_PATH = fullfile(options.PREPROCESSED_PATH);
% path to the intermediate results after running HC-Search
if ~isfield(options, 'HC_INTERMEDIATE_DETECTION_RESULTS_PATH')
    options.HC_INTERMEDIATE_DETECTION_RESULTS_PATH = fullfile(options.TEMP_PATH, 'hc_results');
end
options.HC_INTERMEDIATE_DETECTION_RESULTS_PATH = fullfile(options.HC_INTERMEDIATE_DETECTION_RESULTS_PATH);
% folder to save detection results
if ~isfield(options, 'DETECTION_RESULTS_FOLDER')
    options.DETECTION_RESULTS_FOLDER = 'detection_results';
end
options.DETECTION_RESULTS_FOLDER = fullfile(options.DETECTION_RESULTS_FOLDER);
% time bound parameter for HC-Search
if ~isfield(options, 'HCSEARCH_TIMEBOUND')
    options.HCSEARCH_TIMEBOUND = 1; % default results in IID classifier
end
% use MPI or not
if ~isfield(options, 'USE_MPI')
    options.USE_MPI = 0; % default 0 does not not use MPI; 1 if use MPI
end
% use MPI or not
if ~isfield(options, 'MPI_NUM_PROCESSORS')
    options.MPI_NUM_PROCESSORS = 2; % default is 2 processors
    % must make sure your computer can handle this number of processes
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
update_progress_indicator('Parsing input file', options);

% get character ID from file name
[~, inFileName, ~] = fileparts(inputPath);
[charID, charName] = get_char_id_from_file_name(inFileName);
charStateNames = containers.Map('KeyType', 'char', 'ValueType', 'any');

% get list of training and test instances
[trainingList, scoringList] = parse_input_file(inputPath);
scoringRange = 1+length(trainingList):length(trainingList)+length(scoringList);

telapsed = toc(tglobalstart);
writelog(log_fid, sprintf('Finished parsing input file. (%.1fs)\n\n', telapsed));

ttelapsed = toc(tglobalstart);
writelog(log_fid, sprintf('Total time elapsed so far: %.1fs\n\n', ttelapsed));

%% ========== preprocess data for HC-Search
tstart = tic;
writelog(log_fid, 'Preprocessing input data...\n');
update_progress_indicator('Preprocessing input data', options);

% extract features, preprocess into data for HC-Search
color2label = containers.Map({0, 255}, {-1, 1});
[allData, charStateNames] = preprocess_avatol(options.DATASET_PATH, options.BASE_PATH, trainingList, scoringList, ...
    charID, charStateNames, color2label, options.PREPROCESSED_PATH);

telapsed = toc(tstart);
writelog(log_fid, sprintf('Finished preprocessing input data. (%.1fs)\n\n', telapsed));

ttelapsed = toc(tglobalstart);
writelog(log_fid, sprintf('Total time elapsed so far: %.1fs\n\n', ttelapsed));

%% ========== call HC-Search
tstart = tic;
writelog(log_fid, 'Running character detection...\n');
update_progress_indicator('Running character detection', options);

cmdlineArgs = sprintf('%s %s %d --learn --infer --prune none --ranker vw --successor flipbit-neighbors --base-path %s', ...
    options.PREPROCESSED_PATH, options.HC_INTERMEDIATE_DETECTION_RESULTS_PATH, options.HCSEARCH_TIMEBOUND, options.BASE_PATH);

MPIString = sprintf('mpiexec -n %d ', options.MPI_NUM_PROCESSORS); 
if ispc
    fprintf('Detected PC. Running HC-Search...\n');
    if options.USE_MPI
        [status, result] = dos([MPIString fullfile(options.BASE_PATH, 'HCSearchMPI') ' ' cmdlineArgs]);
    else
        [status, result] = dos([fullfile(options.BASE_PATH, 'HCSearch') ' ' cmdlineArgs]);
    end
else
    fprintf('Detected Unix. Running HC-Search...\n');
    if options.USE_MPI
        [status, result] = unix([MPIString fullfile(options.BASE_PATH, 'HCSearchMPI') ' ' cmdlineArgs]);
    else
        [status, result] = unix([fullfile(options.BASE_PATH, 'HCSearch') ' ' cmdlineArgs]);
    end
end
fprintf('status=\n\n%d\n\n', status);
fprintf('result=\n\n%s\n\n', result);

telapsed = toc(tstart);
writelog(log_fid, sprintf('Finished running character detection. (%.1fs)\n\n', telapsed));

ttelapsed = toc(tglobalstart);
writelog(log_fid, sprintf('Total time elapsed so far: %.1fs\n\n', ttelapsed));

%% ========== postprocess data for character scoring
tstart = tic;
writelog(log_fid, 'Running detection post-process...\n');
update_progress_indicator('Running detection post-process', options);

% postprocess
allData = postprocess_avatol(allData, fullfile(options.HC_INTERMEDIATE_DETECTION_RESULTS_PATH, 'results'), ...
    options.HCSEARCH_TIMEBOUND);

telapsed = toc(tstart);
writelog(log_fid, sprintf('Finished running detection post-process. (%.1fs)\n\n', telapsed));

ttelapsed = toc(tglobalstart);
writelog(log_fid, sprintf('Total time elapsed so far: %.1fs\n\n', ttelapsed));

%% ========== character scoring
tstart = tic;
writelog(log_fid, 'Running character scoring...\n');
update_progress_indicator('Running character scoring', options);

if is_absolute_path(options.DETECTION_RESULTS_FOLDER)
    detectionPath = options.DETECTION_RESULTS_FOLDER;
else
    detectionPath = fullfile(options.DATASET_PATH, options.DETECTION_RESULTS_FOLDER);
end
if ~exist(detectionPath, 'dir')
    fprintf('Creating detection folder: %s\n', detectionPath);
    mkdir(detectionPath);
end

scoringCnt = 1;
newScoringList = {};
nonScoringCnt = 1;
nonScoringList = {};
OFFSET = length(trainingList);
for i = scoringRange
    ttstart = tic;
    fprintf('Scoring image %d...\n', i);
    
    % perform character scoring
    [charState, scoringSuccess] = score_basal_texture(allData{i});
    
    if scoringSuccess
        newScoringList{scoringCnt} = scoringList{i - OFFSET};
        newScoringList{scoringCnt}.charState = charState;

        % save detection polygon
        fprintf('\tSaving detection polygon...\n');
        mediaID = get_media_id_from_path_to_media(newScoringList{scoringCnt}.pathToMedia);
        detectionFile = sprintf('%s_%s.txt', charID, mediaID);
        pathToDetection = fullfile(options.DATASET_PATH, options.DETECTION_RESULTS_FOLDER, detectionFile);
        convert_detection_to_annotation(pathToDetection, allData{i}, charID, charName, charState, charStateNames(charState));

        % save scores
        fprintf('\tSaving scores...\n');
        shortenPathToDetection = fullfile(options.DETECTION_RESULTS_FOLDER, detectionFile);
        newScoringList{scoringCnt}.pathToDetection = shortenPathToDetection;
        
        scoringCnt = scoringCnt + 1;
        
        ttelapsed = toc(ttstart);
        fprintf('\tDone scoring image %i. (%.1fs)\n', i, ttelapsed);
    else
        nonScoringList{nonScoringCnt} = struct;
        nonScoringList{nonScoringCnt}.pathToMedia = scoringList{i - OFFSET}.pathToMedia;
        
        nonScoringCnt = nonScoringCnt + 1;
        
        ttelapsed = toc(ttstart);
        fprintf('\tCould not score image %i. (%.1fs)\n', i, ttelapsed);
    end
end

telapsed = toc(tstart);
writelog(log_fid, sprintf('Finished running character scoring. (%.1fs)\n\n', telapsed));

ttelapsed = toc(tglobalstart);
writelog(log_fid, sprintf('Total time elapsed so far: %.1fs\n\n', ttelapsed));

%% ========== save scores
tstart = tic;
writelog(log_fid, 'Saving scores...\n');
update_progress_indicator('Saving scores', options);

% save scores
write_scores(outputPath, trainingList, newScoringList, nonScoringList);

telapsed = toc(tstart);
writelog(log_fid, sprintf('Finished saving scores. (%.1fs)\n', telapsed));

ttelapsed = toc(tglobalstart);
writelog(log_fid, sprintf('Total time elapsed so far: %.1fs\n\n', ttelapsed));

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

function mediaID = get_media_id_from_path_to_media(pathToMedia)

[~, parsed, ~] = fileparts(pathToMedia);
parsed = textscan(parsed, '%s', 'delimiter', '_');
mediaID = parsed{1}{1};

end

function update_progress_indicator(message, options)

if isfield(options, 'PROGRESS_INDICATOR')
    options.PROGRESS_INDICATOR.setStatus(message);
end

end