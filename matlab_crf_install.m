%% Script to install dependencies for the first time.
%% ***Only run once for installation.***
%% However, run again if you need to refresh your installation.

% Force compilation of mex files, especially for Windows
% Change below: 0 = do not force, 1 = force compile
FORCE_COMPILE = 0;

%% NO NEED TO EDIT ANYTHING BELOW
%% EDIT SETTINGS IN matlab_crf_settings.m

INITIAL_INSTALL = 1;
matlab_crf_setup_include_paths;