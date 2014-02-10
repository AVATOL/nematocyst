function plot_anytime_curves( evaluate, MODE )
%PLOT_ANYTIME_CURVES Plot anytime curves.
%   
%   evaluate:   map containing evaluation structures
%   MODE:       0 = binary evaluation, 1 = macro measures, 2 = micro measures

narginchk(1, 2);

%% mode 0 = binary, 1 = macro, 2 = micro
if nargin < 2
    MODE = 1;
end

%% search types
searchTypesCollection = cell(1, 4);
searchTypesCollection{1} = 'hc';
searchTypesCollection{2} = 'hl';
searchTypesCollection{3} = 'lc';
searchTypesCollection{4} = 'll';

searchTypesAvailable = [];
for s = 1:length(searchTypesCollection)
    searchType = searchTypesCollection{s};
    
    if ~isKey(evaluate, searchType)
        continue;
    end
    
    searchTypesAvailable = [searchTypesAvailable; s];
    timeRange = evaluate(searchType).timeRange;
end % search types

avgPrecMat = zeros(length(timeRange), length(searchTypesAvailable));
stdPrecMat = zeros(length(timeRange), length(searchTypesAvailable));
avgRecMat = zeros(length(timeRange), length(searchTypesAvailable));
stdRecMat = zeros(length(timeRange), length(searchTypesAvailable));
avgF1Mat = zeros(length(timeRange), length(searchTypesAvailable));
stdF1Mat = zeros(length(timeRange), length(searchTypesAvailable));

legendLabels = cell(length(searchTypesAvailable), 1);
for i = 1:length(searchTypesAvailable)
    searchType = searchTypesCollection{searchTypesAvailable(i)};
    evaluateType = evaluate(searchType);
    
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
    
    legendLabels{i} = sprintf('%s', upper(searchTypesCollection{searchTypesAvailable(i)}));
end

timeBoundsMat = repmat(timeRange, length(searchTypesAvailable), 1)';

errorbar(timeBoundsMat,...
avgPrecMat,...
stdPrecMat);
hold on;
title('Precision vs. Time Bound');
xlabel('Time Bound');
ylabel('Precision');
legend(legendLabels);
hold off;
pause;

errorbar(timeBoundsMat,...
    avgRecMat,...
    stdRecMat);
hold on;
title('Recall vs. Time Bound');
xlabel('Time Bound');
ylabel('Recall');
legend(legendLabels);
hold off;
pause;

errorbar(timeBoundsMat,... 
    avgF1Mat,... 
    stdF1Mat);
hold on;
title('F1 vs. Time Bound');
xlabel('Time Bound');
ylabel('F1');
legend(legendLabels);
hold off;
pause;

end

