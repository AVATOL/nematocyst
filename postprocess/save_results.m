function [ results ] = save_results( preprocessedDir, resultsDir, timeRange, foldRange, searchTypes, splitsName )
%SAVE_RESULTS Save results in preparation for evaluation.
%
%	preprocessedDir:	folder path containing preprocessed data
%                           e.g. 'DataPreprocessed/SomeDataset'
%	resultsDir:         folder path containing HC-Search results
%                           e.g. 'Results/SomeExperiment'
%   timeRange:          range of time bound
%   foldRange:          range of folds
%   searchTypes:        list of search types 1 = HC, 2 = HL, 3 = LC, 4 = LL
%   splitsName:         (optional) alternate name to splits folder

narginchk(4, 6);

if nargin < 6
    splitsName = 'splits';
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
testSplitsFile = [preprocessedDir '/' splitsName '/Test.txt'];
fid = fopen(testSplitsFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
testFiles = list{1,1};

%% prepare output data structure
results = containers.Map;

%% for each search type
for s = searchTypes
    searchType = searchTypesCollection{s};
    fprintf('On search type %s...\n', searchType);
    
    %% prepare data structure
    struct = cell(length(foldRange), length(testFiles), length(timeRange));
    
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
            
            %% for each time step
            for t = 1:length(timeRange)
                timeStep = timeRange(t);
                fprintf('\t\t\tOn time step %d...\n', timeStep);

                edgesFileName = sprintf('edges_%s_test_time%d_fold%d_%s.txt', searchType, timeStep, fold, fileName);
                nodesFileName = sprintf('nodes_%s_test_time%d_fold%d_%s.txt', searchType, timeStep, fold, fileName);
                
                cutsPath = [resultsDir '/results/' edgesFileName];
                nodesPath = [resultsDir '/results/' nodesFileName];

                if ~exist(cutsPath, 'file') || ~exist(nodesPath, 'file')
                    continue;
                end

                %% read nodes
                [inferLabels, ~] = libsvmread(nodesPath);
                
                %% read edges
                cutMat = dlmread(cutsPath);
                cutMat = spconvert(cutMat);
                cutMat = full(cutMat);

                nNodes = length(truthLabels);
                [h, w] = size(cutMat);
                cutMat = [cutMat zeros(h, nNodes-w)];
                [h, w] = size(cutMat);
                cutMat = [cutMat; zeros(nNodes-h, w)];
                
                %% save to data structure
                struct{fd, f, t}.infer = inferLabels;
                struct{fd, f, t}.truth = truthLabels;
                struct{fd, f, t}.cuts = cutMat;
                struct{fd, f, t}.filename = fileName;
            end % time range
        end % files
    end % fold
    
    %% add
    results(searchType) = struct;
end % search type

end
