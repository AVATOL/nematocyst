function plot_compare_curves( evaluate, labels, searchTypeIndices, plotTitle, MODE, configFlag )
%PLOT_COMPARE_CURVES Compare anytime curve plots.
%   
%   evaluate:           cells contain a map that contains evaluation structures
%   labels:             cells containing strings labeling the corresponding experiments
%   searchTypeIndices:  indices for search type
%   plotTitle:          title of plot
%   MODE:               0 = binary evaluation, 1 = macro measures, 2 = micro measures
%   configFlag:         flag for configuration options

narginchk(3, 6);

if nargin < 6
    configFlag = 1; % 1 = Stanford, 2 = Nematocysts
end
%% mode 0 = binary, 1 = macro, 2 = micro
if nargin < 5
    MODE = 1;
end

if configFlag == 1
    SHOW_PREC = 0;
    SHOW_REC = 0;
    SHOW_F1 = 0;
    SHOW_ACC = 1;
elseif configFlag == 2
    SHOW_PREC = 1;
    SHOW_REC = 1;
    SHOW_F1 = 1;
    SHOW_ACC = 0;
end

%% search types
searchTypesCollection = cell(1, 4);
searchTypesCollection{1} = 'hc';
searchTypesCollection{2} = 'hl';
searchTypesCollection{3} = 'lc';
searchTypesCollection{4} = 'll';

if nargin < 4
    plotTitle = 'Untitled';
end

% get minimum time range if time ranges are different
temp = evaluate{1};
timeRange = temp(searchTypesCollection{searchTypeIndices(1)}).timeRange;
for i = 2:length(evaluate)
    temp = evaluate{i};
    timeRangeTemp = temp(searchTypesCollection{searchTypeIndices(1)}).timeRange;
    if length(timeRangeTemp) < length(timeRange)
        timeRange = timeRangeTemp;
    end
end

avgPrecMat = zeros(length(timeRange), length(evaluate));
stdPrecMat = zeros(length(timeRange), length(evaluate));
avgRecMat = zeros(length(timeRange), length(evaluate));
stdRecMat = zeros(length(timeRange), length(evaluate));
avgF1Mat = zeros(length(timeRange), length(evaluate));
stdF1Mat = zeros(length(timeRange), length(evaluate));
if configFlag ~= 2
    avgHammingMat = zeros(length(timeRange), length(evaluate));
    stdHammingMat = zeros(length(timeRange), length(evaluate));
end

legendLabels = cell(length(evaluate), 1);
for i = 1:length(evaluate)
    searchType = searchTypesCollection{searchTypeIndices(i)};
    evaluateInstance = evaluate{i};
    evaluateType = evaluateInstance(searchType);
    
    if MODE == 0
        avgPrecMat(:, i) = evaluateType.binary_avgprec(timeRange+1)';
        stdPrecMat(:, i) = evaluateType.binary_stdprec(timeRange+1)';
        avgRecMat(:, i) = evaluateType.binary_avgrec(timeRange+1)';
        stdRecMat(:, i) = evaluateType.binary_stdrec(timeRange+1)';
        avgF1Mat(:, i) = evaluateType.binary_avgf1(timeRange+1)';
        stdF1Mat(:, i) = evaluateType.binary_stdf1(timeRange+1)';
    elseif MODE == 1
        avgPrecMat(:, i) = evaluateType.avgmacroprec(timeRange+1)';
        stdPrecMat(:, i) = evaluateType.stdmacroprec(timeRange+1)';
        avgRecMat(:, i) = evaluateType.avgmacrorec(timeRange+1)';
        stdRecMat(:, i) = evaluateType.stdmacrorec(timeRange+1)';
        avgF1Mat(:, i) = evaluateType.avgmacrof1(timeRange+1)';
        stdF1Mat(:, i) = evaluateType.stdmacrof1(timeRange+1)';
    elseif MODE == 2
        avgPrecMat(:, i) = evaluateType.avgmicroprec(timeRange+1)';
        stdPrecMat(:, i) = evaluateType.stdmicroprec(timeRange+1)';
        avgRecMat(:, i) = evaluateType.avgmicrorec(timeRange+1)';
        stdRecMat(:, i) = evaluateType.stdmicrorec(timeRange+1)';
        avgF1Mat(:, i) = evaluateType.avgmicrof1(timeRange+1)';
        stdF1Mat(:, i) = evaluateType.stdmicrof1(timeRange+1)';
    end
    
    if configFlag ~= 2
        avgHammingMat(:, i) = evaluateType.avghamming(timeRange+1)';
        stdHammingMat(:, i) = evaluateType.stdhamming(timeRange+1)';
    end
    
    legendLabels{i} = sprintf('%s', labels{i});
end

timeBoundsMat = repmat(timeRange, length(evaluate), 1)';

if SHOW_PREC
    figure;
    errorbar(timeBoundsMat,...
    avgPrecMat,...
    stdPrecMat);
    hold on;
    title(sprintf('%s: Precision vs. Time Bound', plotTitle));
    xlabel('Time Bound');
    ylabel('Precision');
    legend(legendLabels);
    hold off;
end

if SHOW_REC
    figure;
    errorbar(timeBoundsMat,...
        avgRecMat,...
        stdRecMat);
    hold on;
    title(sprintf('%s: Recall vs. Time Bound', plotTitle));
    xlabel('Time Bound');
    ylabel('Recall');
    legend(legendLabels);
    hold off;
end

if SHOW_F1
    figure;
    errorbar(timeBoundsMat,... 
        avgF1Mat,... 
        stdF1Mat);
    hold on;
    title(sprintf('%s: F1 vs. Time Bound', plotTitle));
    xlabel('Time Bound');
    ylabel('F1');
    legend(legendLabels);
    hold off;
end

if SHOW_ACC
    figure;
    errorbar(timeBoundsMat,... 
        avgHammingMat,... 
        stdHammingMat);
    hold on;
    title(sprintf('%s: Hamming Accuracy vs. Time Bound', plotTitle));
    xlabel('Time Bound');
    ylabel('Hamming Accuracy');
    legend(legendLabels);
    hold off;
end

end

