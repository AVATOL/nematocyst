%% script to make sure all paths are set up

%% initialize
[scripts_path, ~, ~] = fileparts(mfilename('fullpath'));

%% check folders
CHECK_FOLDERS = {'external', ['external' filesep 'Eigen'], ...
    ['external' filesep 'liblinear'], ['external' filesep 'libsvm'], ...
    ['external' filesep 'vowpal_wabbit']};

%% check folders exist otherwise display error
for i = 1:length(CHECK_FOLDERS)
    folderCheck = [scripts_path filesep CHECK_FOLDERS{i}];
    if ~exist(folderCheck, 'dir')
       error('folder "%s" does not exist!', folderCheck); 
    end
end

%% set include paths for MATLAB
INCLUDE_FOLDERS = {'avatol_system'; 'character_scoring'; 'postprocess'; 'preprocess'; 'scripts'; 'utilities'};

%% add folders to include path in MATLAB
for i = 1:length(INCLUDE_FOLDERS)
    addpath(genpath([scripts_path filesep INCLUDE_FOLDERS{i}]));
end

%% add VLFeat to path
if exist('vl_version') ~= 3;
    folderCheck = [scripts_path filesep 'external' filesep 'vlfeat'];
    if exist(folderCheck, 'dir')
        run([folderCheck filesep 'toolbox' filesep 'vl_setup']);
    else
        error('cannot install vlfeat: folder "%s" does not exist!', folderCheck); 
    end
end

%% cleanup
clearvars;