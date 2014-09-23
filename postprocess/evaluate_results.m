function [ evaluate ] = evaluate_results( preprocessedDir, resultsDir, timeRange, foldRange, searchTypes, splitsName, configFlag )
%EVALUATE_RESULTS Evaluate results in preparation for plotting anytime
%curves.
%
%	preprocessedDir:	folder path containing preprocessed data
%                           e.g. 'DataPreprocessed/SomeDataset'
%	resultsDir:         folder path containing HC-Search results
%                           e.g. 'Results/SomeExperiment'
%   timeRange:          range of time bound
%   foldRange:          range of folds
%   searchTypes:        list of search types 1 = HC, 2 = HL, 3 = LC, 4 = LL
%   splitsName:         (optional) alternate name to splits folder and file
%   configFlag:         flag for configuration options

narginchk(4, 7);

if nargin < 7
    configFlag = 1; % 1 = Stanford, 2 = Nematocysts
end
if nargin < 6
    splitsName = 'splits/Test.txt';
end
if nargin < 5
    searchTypes = [1 2 3 4];
end

%% search types
searchTypesCollection = cell(1, 5);
searchTypesCollection{1} = 'hc';
searchTypesCollection{2} = 'hl';
searchTypesCollection{3} = 'lc';
searchTypesCollection{4} = 'll';
searchTypesCollection{5} = 'rl';

%% test files
testSplitsFile = [preprocessedDir '/' splitsName];
fid = fopen(testSplitsFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
testFiles = list{1,1};

%% get all classes
classes = get_classes_from_metafile(preprocessedDir);
if configFlag == 1
    IGNORE_CLASSES = 0;                 % Stanford uses 0 for ignore
    BINARY_FOREGROUND_CLASS_INDEX = 8;  % 8 means foreground class in Stanford
elseif configFlag == 2
    IGNORE_CLASSES = [];                % [] means nothing to ignore
    BINARY_FOREGROUND_CLASS_INDEX = 2;  % [-1, 1] means index 2 is foreground
else
    error('unknown config');
end

%% prepare output data structure
evaluate = containers.Map;

%% for each search type
for s = searchTypes
    searchType = searchTypesCollection{s};
    fprintf('On search type %s...\n', searchType);
    
    %% prepare data structure
    stat.timeRange = timeRange;
    stat.classes = classes;
    
    stat.tp = zeros(length(foldRange), length(timeRange), length(classes));
    stat.fp = zeros(length(foldRange), length(timeRange), length(classes));
    stat.tn = zeros(length(foldRange), length(timeRange), length(classes));
    stat.fn = zeros(length(foldRange), length(timeRange), length(classes));

    stat.prec = zeros(length(foldRange), length(timeRange), length(classes));
    stat.rec = zeros(length(foldRange), length(timeRange), length(classes));
    stat.f1 = zeros(length(foldRange), length(timeRange), length(classes));
    
    stat.macroprec = zeros(length(foldRange), length(timeRange));
    stat.macrorec = zeros(length(foldRange), length(timeRange));
    stat.macrof1 = zeros(length(foldRange), length(timeRange));

    stat.microprec = zeros(length(foldRange), length(timeRange));
    stat.microrec = zeros(length(foldRange), length(timeRange));
    stat.microf1 = zeros(length(foldRange), length(timeRange));
    
    stat.binary_prec = zeros(length(foldRange), length(timeRange));
    stat.binary_rec = zeros(length(foldRange), length(timeRange));
    stat.binary_f1 = zeros(length(foldRange), length(timeRange));
    
    if configFlag ~= 2
        stat.numcorrect = zeros(length(foldRange), length(timeRange));
        stat.totals = zeros(length(foldRange), length(timeRange));
        stat.hamming = zeros(length(foldRange), length(timeRange));
    end
    
    stat.avgmacroprec = zeros(1, length(timeRange));
    stat.avgmacrorec = zeros(1, length(timeRange));
    stat.avgmacrof1 = zeros(1, length(timeRange));
    stat.stdmacroprec = zeros(1, length(timeRange));
    stat.stdmacrorec = zeros(1, length(timeRange));
    stat.stdmacrof1 = zeros(1, length(timeRange));
    
    stat.avgmicroprec = zeros(1, length(timeRange));
    stat.avgmicrorec = zeros(1, length(timeRange));
    stat.avgmicrof1 = zeros(1, length(timeRange));
    stat.stdmicroprec = zeros(1, length(timeRange));
    stat.stdmicrorec = zeros(1, length(timeRange));
    stat.stdmicrof1 = zeros(1, length(timeRange));
    
    stat.binary_avgprec = zeros(1, length(timeRange));
    stat.binary_avgrec = zeros(1, length(timeRange));
    stat.binary_avgf1 = zeros(1, length(timeRange));
    stat.binary_stdprec = zeros(1, length(timeRange));
    stat.binary_stdrec = zeros(1, length(timeRange));
    stat.binary_stdf1 = zeros(1, length(timeRange));
    
    if configFlag ~= 2
        stat.avghamming = zeros(1, length(timeRange));
        stat.stdhamming = zeros(1, length(timeRange));
    end
    
    %% for each fold
    for fd = 1:length(foldRange)
        fold = foldRange(fd);
        fprintf('\tOn fold %d...\n', fold);

        %% for each file
        for f = 1:length(testFiles)
            fileName = testFiles{f};
            fprintf('\t\tOn file %s...\n', fileName);

            %% read truth nodes
            truthNodesPath = [preprocessedDir '/nodes/' fileName '.txt'];
            [truthLabels, ~] = libsvmread(truthNodesPath);
            
            %% read segments
            segmentsPath = [preprocessedDir '/segments/' fileName '.txt'];
            segments = dlmread(segmentsPath);
            
            %% read truth labeling
            fullTruthPath = [preprocessedDir '/groundtruth/' fileName '.txt'];
            fullTruth = dlmread(fullTruthPath);
            
            %% for each time step
            prev = '';
            for t = 1:length(timeRange)
                timeStep = timeRange(t);
                fprintf('\t\t\tOn time step %d...\n', timeStep);

                nodesFileName = sprintf('nodes_%s_test_time%d_fold%d_%s.txt', searchType, timeStep, fold, fileName);
                nodesPath = [resultsDir '/results/' nodesFileName];

                if t ~= 1 && ~exist(nodesPath, 'file')
                    nodesPath = prev;
                end

                %% read nodes
                [inferLabels, ~] = libsvmread(nodesPath);
                
                %% read inference on pixel level
                if configFlag ~= 2
                    inferPixels = infer_pixels(inferLabels, segments);
                end
                
                %% calculations...
                for c = 1:length(classes)
                    classLabel = classes(c);
                    
                    [tp, fp, tn, fn] = calculate(inferLabels, truthLabels, classLabel, IGNORE_CLASSES);
                    
                    stat.tp(fd, t, c) = stat.tp(fd, t, c) + tp;
                    stat.fp(fd, t, c) = stat.fp(fd, t, c) + fp;
                    stat.tn(fd, t, c) = stat.tn(fd, t, c) + tn;
                    stat.fn(fd, t, c) = stat.fn(fd, t, c) + fn;
                end % classes
                
                if configFlag ~= 2
                    stat.numcorrect(fd, t) = stat.numcorrect(fd, t) + sum(sum(double(inferPixels == fullTruth)));
                    stat.totals(fd, t) = stat.totals(fd, t) + numel(fullTruth);
                end
                
                prev = nodesPath;
            end % time range
        end % files
    end % fold

    %% calculate hamming
    if configFlag ~= 2
        stat.hamming = stat.numcorrect/stat.totals(fd, t);
    end
    
    %% calculate non-macro/micro measures
    stat.prec = stat.tp ./ (stat.tp + stat.fp);
    stat.rec = stat.tp ./ (stat.tp + stat.fn);
    stat.f1 = 2*(stat.prec .* stat.rec) ./ (stat.prec + stat.rec);
    
    stat.prec(isnan(stat.prec)) = 0;
    stat.rec(isnan(stat.rec)) = 0;
    stat.f1(isnan(stat.f1)) = 0;
    
    %% calculate macro measures
    stat.macroprec = mean(stat.prec, 3);
    stat.macrorec = mean(stat.rec, 3);
    stat.macrof1 = mean(stat.f1, 3);
    
    %% calculate micro measures
    microtp = sum(stat.tp, 3);
    microfp = sum(stat.fp, 3);
    microfn = sum(stat.fn, 3);
    
    stat.microprec = microtp ./ (microtp + microfp);
    stat.microrec = microtp ./ (microtp + microfn);
    stat.microf1 = 2*(stat.microprec .* stat.microrec) ./ (stat.microprec + stat.microrec);
    
    stat.microprec(isnan(stat.microprec)) = 0;
    stat.microrec(isnan(stat.microrec)) = 0;
    stat.microf1(isnan(stat.microf1)) = 0;
    
    %% calculate binary measures
    stat.binary_prec = stat.prec(:, :, BINARY_FOREGROUND_CLASS_INDEX);
    stat.binary_rec = stat.rec(:, :, BINARY_FOREGROUND_CLASS_INDEX);
    stat.binary_f1 = stat.f1(:, :, BINARY_FOREGROUND_CLASS_INDEX);
    
    %% calculate average and standard deviation across folds
    stat.avgmacroprec = mean(stat.macroprec, 1);
    stat.avgmacrorec = mean(stat.macrorec, 1);
    stat.avgmacrof1 = mean(stat.macrof1, 1);
    stat.stdmacroprec = std(stat.macroprec, 0, 1);
    stat.stdmacrorec = std(stat.macrorec, 0, 1);
    stat.stdmacrof1 = std(stat.macrof1, 0, 1);
    
    stat.avgmicroprec = mean(stat.microprec, 1);
    stat.avgmicrorec = mean(stat.microrec, 1);
    stat.avgmicrof1 = mean(stat.microf1, 1);
    stat.stdmicroprec = std(stat.microprec, 0, 1);
    stat.stdmicrorec = std(stat.microrec, 0, 1);
    stat.stdmicrof1 = std(stat.microf1, 0, 1);
    
    stat.binary_avgprec = mean(stat.binary_prec, 1);
    stat.binary_avgrec = mean(stat.binary_rec, 1);
    stat.binary_avgf1 = mean(stat.binary_f1, 1);
    stat.binary_stdprec = std(stat.binary_prec, 0, 1);
    stat.binary_stdrec = std(stat.binary_rec, 0, 1);
    stat.binary_stdf1 = std(stat.binary_f1, 0, 1);
    
    if configFlag ~= 2
        stat.avghamming = mean(stat.hamming, 1);
        stat.stdhamming = std(stat.hamming, 0, 1);
    end
    
    %% add
    evaluate(searchType) = stat;
end % search type

save([resultsDir '/evaluate.mat'], 'evaluate');

end

function [classSet] = get_classes(fileNameSet, preprocessedDir)

classSet = [];

for f = 1:length(fileNameSet)
    fileName = fileNameSet{f};
    fprintf('On file %s...\n', fileName);

    %% read truth nodes
    truthNodesPath = [preprocessedDir '/nodes/' fileName '.txt'];
    [truthLabels, ~] = libsvmread(truthNodesPath);
    
    classSet = union(classSet, truthLabels);
end

classSet = sort(classSet);

end

function [classSet] = get_classes_from_metafile(preprocessedDir)

fileData = fileread([preprocessedDir '/metadata.txt']);
% following assumes classes appears before backgroundclasses appears at end
beginIndex = regexp(fileData, 'classes=')+length('classes=');
endIndex = regexp(fileData, 'backgroundclasses=')-1;
stringData = fileData(beginIndex:endIndex);
stringArray = textscan(stringData, '%s', 'delimiter', ',');
labelStrings = stringArray{1};
classSet = zeros(1, length(labelStrings));
for i = 1:length(labelStrings)
    string = labelStrings{i};
    num = str2num(string);
    classSet(i) = num;
end

end

function [tp, fp, tn, fn] = calculate(inferLabels, truthLabels, classLabel, IGNORE_CLASSES)

for ignoreClass = IGNORE_CLASSES
    ignoreIndices = find(truthLabels == ignoreClass);
    inferLabels(ignoreIndices) = [];
    truthLabels(ignoreIndices) = [];
end

tp = sum(double((inferLabels == classLabel) & (truthLabels == classLabel)));
fp = sum(double((inferLabels == classLabel) & (truthLabels ~= classLabel)));
tn = sum(double((inferLabels ~= classLabel) & (truthLabels ~= classLabel)));
fn = sum(double((inferLabels ~= classLabel) & (truthLabels == classLabel)));

end

function [inferPixels] = infer_pixels(inferLabels, segments)

inferPixels = zeros(size(segments));
nNodes = length(inferLabels);

for i = 1:nNodes
    temp = segments;
    temp(temp ~= i) = 0;
    temp(temp == i) = inferLabels(i);
    
    inferPixels = inferPixels + temp;
end

end