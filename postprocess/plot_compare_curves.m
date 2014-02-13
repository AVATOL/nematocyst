function plot_compare_curves( evaluate, labels, searchTypeIndices, plotTitle, MODE )
%PLOT_COMPARE_CURVES Compare anytime curve plots.
%   
%   evaluate:           cells contain a map that contains evaluation structures
%   labels:             cells containing strings labeling the corresponding experiments
%   searchTypeIndex:    index for search type
%   plotTitle:          title of plot
%   MODE:               0 = binary evaluation, 1 = macro measures, 2 = micro measures

narginchk(3, 5);

%% mode 0 = binary, 1 = macro, 2 = micro
if nargin < 5
    MODE = 1;
end

%% search types
searchTypesCollection = cell(1, 5);
searchTypesCollection{1} = 'hc';
searchTypesCollection{2} = 'hl';
searchTypesCollection{3} = 'lc';
searchTypesCollection{4} = 'll';
searchTypesCollection{5} = 'rl';

if nargin < 4
    plotTitle = 'Untitled';
end

temp = evaluate{1};
timeRange = temp(searchTypesCollection{searchTypeIndices{1}}).timeRange;

avgPrecMat = zeros(length(timeRange), length(evaluate));
stdPrecMat = zeros(length(timeRange), length(evaluate));
avgRecMat = zeros(length(timeRange), length(evaluate));
stdRecMat = zeros(length(timeRange), length(evaluate));
avgF1Mat = zeros(length(timeRange), length(evaluate));
stdF1Mat = zeros(length(timeRange), length(evaluate));

legendLabels = cell(length(evaluate), 1);
for i = 1:length(evaluate)
    searchType = searchTypesCollection{searchTypeIndices{i}};
    evaluateInstance = evaluate{i};
    evaluateType = evaluateInstance(searchType);
    
    if MODE == 0
        avgPrecMat(:, i) = evaluateType.binary_avgprec';
        stdPrecMat(:, i) = evaluateType.binary_stdprec';
        avgRecMat(:, i) = evaluateType.binary_avgrec';
        stdRecMat(:, i) = evaluateType.binary_stdrec';
        avgF1Mat(:, i) = evaluateType.binary_avgf1';
        stdF1Mat(:, i) = evaluateType.binary_stdf1';
    elseif MODE == 1
        avgPrecMat(:, i) = evaluateType.avgmacroprec';
        stdPrecMat(:, i) = evaluateType.stdmacroprec';
        avgRecMat(:, i) = evaluateType.avgmacrorec';
        stdRecMat(:, i) = evaluateType.stdmacrorec';
        avgF1Mat(:, i) = evaluateType.avgmacrof1';
        stdF1Mat(:, i) = evaluateType.stdmacrof1';
    elseif MODE == 2
        avgPrecMat(:, i) = evaluateType.avgmicroprec';
        stdPrecMat(:, i) = evaluateType.stdmicroprec';
        avgRecMat(:, i) = evaluateType.avgmicrorec';
        stdRecMat(:, i) = evaluateType.stdmicrorec';
        avgF1Mat(:, i) = evaluateType.avgmicrof1';
        stdF1Mat(:, i) = evaluateType.stdmicrof1';
    end
    
    legendLabels{i} = sprintf('%s', labels{i});
end

timeBoundsMat = repmat(timeRange, length(evaluate), 1)';

errorbar(timeBoundsMat,...
avgPrecMat,...
stdPrecMat);
hold on;
title(sprintf('%s: Precision vs. Time Bound', plotTitle));
xlabel('Time Bound');
ylabel('Precision');
legend(legendLabels);
hold off;
pause;

errorbar(timeBoundsMat,...
    avgRecMat,...
    stdRecMat);
hold on;
title(sprintf('%s: Recall vs. Time Bound', plotTitle));
xlabel('Time Bound');
ylabel('Recall');
legend(legendLabels);
hold off;
pause;

errorbar(timeBoundsMat,... 
    avgF1Mat,... 
    stdF1Mat);
hold on;
title(sprintf('%s: F1 vs. Time Bound', plotTitle));
xlabel('Time Bound');
ylabel('F1');
legend(legendLabels);
hold off;
pause;

end

