%% set include paths for MATLAB
INCLUDE_FOLDERS = {'avatol_system'; 'character_scoring'; 'postprocess'; 'preprocess'; 'scripts'; 'utilities'};

%% add folders to include path in MATLAB
[scripts_path, ~, ~] = fileparts(mfilename('fullpath'));
for i = 1:length(INCLUDE_FOLDERS)
    addpath(genpath([scripts_path filesep INCLUDE_FOLDERS{i}]));
end

clearvars;