%% change paths and parameters below

% path to original images containing .jpg files
allDataPath = ['Data' filesep 'SBD'];

% path to output file
outputPath = ['Data' filesep 'SBD_Pre'];

% specify training, validation and test sets
% can also generate random subsets and set here
trainRange = 1:520;
validRange = 521:570;
testRange = 570:715;

%% done
load([allDataPath filesep 'allData.mat']);
allData = preprocess_alldata(allData, outputPath, trainRange, validRange, testRange);