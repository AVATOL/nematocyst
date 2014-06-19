%% change paths and parameters below

% path to original images containing .jpg files
imagesPath = ['Data' filesep 'Nematocysts' filesep 'Images'];

% path to groundtruth annotations containing .jpg files
labelsPath = ['Data' filesep 'Nematocysts' filesep 'Groundtruth'];

% maps color of groundtruth mask to integer class label
% binary groundtruth mask: background, foreground
color2label = containers.Map({0, 255}, {-1, 1});
% multiclass groundtruth mask: clutter, background, basal tubule, capsule, distal tubule
%color2label = containers.Map({32 0 255 128 64}, {-1, 0, 1, 2, 3}); %

% path to splits files
% folder must contain Train.txt, Validation.txt and Test.txt
splitsPath = ['Data' filesep 'Nematocysts' filesep 'SplitsDemo'];

% path to output file
outputPath = ['Data' filesep 'Nemato_Pre_BinaryDemo'];

%% done
allData = preprocess_grid_grayscale(imagesPath, labelsPath, color2label, splitsPath, outputPath);