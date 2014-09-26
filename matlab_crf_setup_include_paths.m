%% Script to install dependencies for the first time.
%% ***Run every time MATLAB starts up.***

current_dir = pwd;
[scripts_path, ~, ~] = fileparts(mfilename('fullpath'));

%% EDIT SETTINGS BELOW IF NECESSARY

%% installation options: 1 = yes, 0 = no
INSTALL_VLFEAT = 1;
INSTALL_LIBLINEAR = 1;
INSTALL_LIBSVM = 1;

%% paths to libraries
EXTERNAL_FOLDER = fullfile('external');
EXTERNAL_PATH = fullfile(scripts_path, EXTERNAL_FOLDER);
VLFEAT_PATH = fullfile(EXTERNAL_PATH, 'vlfeat');
LIBLINEAR_PATH = fullfile(EXTERNAL_PATH, 'liblinear', 'matlab');
LIBSVM_PATH = fullfile(EXTERNAL_PATH, 'libsvm', 'matlab');
LIBLINEAR_PATH_WINDOWS = fullfile(EXTERNAL_PATH, 'liblinear', 'windows');
LIBSVM_PATH_WINDOWS = fullfile(EXTERNAL_PATH, 'libsvm', 'windows');

%% set include paths for MATLAB
INCLUDE_FOLDERS = {'avatol_system'; 'character_scoring'; ...
    'postprocess'; 'preprocess'; 'scripts'; 'utilities'};

%% END OF EDIT SECTION
%% NO NEED TO EDIT ANYTHING BELOW

if ~exist('INITIAL_INSTALL', 'var')
    INITIAL_INSTALL = 1;
end
if ~exist('FORCE_COMPILE', 'var')
    FORCE_COMPILE = 0;
end

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
    if FORCE_COMPILE || (INITIAL_INSTALL && ~ispc)
        fprintf('Compiling LIBLINEAR...');
        cd(LIBLINEAR_PATH);
        make;
        cd(current_dir);
        fprintf('done.\n');
    end
    fprintf('Adding LIBLINEAR to include path...');
    addpath(LIBLINEAR_PATH);
    if ispc
       addpath(LIBLINEAR_PATH_WINDOWS); 
    end
    fprintf('done.\n');
end

%% install libsvm
if INSTALL_LIBSVM
    if FORCE_COMPILE || (INITIAL_INSTALL && ~ispc)
        fprintf('Compiling LIBSVM...');
        cd(LIBSVM_PATH);
        make;
        cd(current_dir);
        fprintf('done.\n');
    end
    fprintf('Adding LIBSVM to include path...');
    addpath(LIBSVM_PATH);
    if ispc
        addpath(LIBSVM_PATH_WINDOWS);
    end
    fprintf('done.\n');
end

%% cleanup
clear