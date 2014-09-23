%% This script installs dependencies and/or adds folders to the path.

current_dir = pwd;
[scripts_path, ~, ~] = fileparts(mfilename('fullpath'));

%% EDIT SETTINGS BELOW IF NECESSARY

%% installation options: 1 = yes, 0 = no
INSTALL_VLFEAT = 1;
INSTALL_LIBLINEAR = 1;
INSTALL_LIBSVM = 1;

%% paths to libraries
EXTERNAL_FOLDER = 'external';
EXTERNAL_PATH = [scripts_path filesep EXTERNAL_FOLDER];
VLFEAT_PATH = [EXTERNAL_PATH filesep 'vlfeat'];
LIBLINEAR_PATH = [EXTERNAL_PATH filesep 'liblinear' filesep 'matlab'];
LIBSVM_PATH = [EXTERNAL_PATH filesep 'libsvm' filesep 'matlab'];

%% set include paths for MATLAB
INCLUDE_FOLDERS = {'avatol_system'; 'character_scoring'; ...
    'postprocess'; 'preprocess'; 'scripts'; 'utilities'};

%% END OF EDIT SECTION
%% NO NEED TO EDIT ANYTHING BELOW

%% add folders to include path in MATLAB
fprintf('Adding essential folders to include path...');
for i = 1:length(INCLUDE_FOLDERS)
    addpath(genpath([scripts_path filesep INCLUDE_FOLDERS{i}]));
end
fprintf('done.\n');

%% install vl feat
if INSTALL_VLFEAT
    fprintf('Installing and adding to path VLFeat...');
    run([VLFEAT_PATH filesep 'toolbox' filesep 'vl_setup']);
    fprintf('done.\n');
    vl_version verbose;
end

%% install liblinear
if INSTALL_LIBLINEAR
    if INITIAL_INSTALL
        fprintf('Installing LIBLINEAR...');
        cd(LIBLINEAR_PATH);
        make;
        cd(current_dir);
        fprintf('done.\n');
    end
    fprintf('Adding LIBLINEAR to include path...');
    addpath(LIBLINEAR_PATH);
    fprintf('done.\n');
end

%% install libsvm
if INSTALL_LIBSVM
    if INITIAL_INSTALL
        fprintf('Installing LIBSVM...');
        cd(LIBSVM_PATH);
        make;
        cd(current_dir);
        fprintf('done.\n');
    end
    fprintf('Adding LIBSVM to include path...');
    addpath(LIBSVM_PATH);
    fprintf('done.\n');
end

%% cleanup
clear