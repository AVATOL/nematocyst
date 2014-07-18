close all;

%% store all the eccentricies and minor/major axis values
data = cell(size(allData));

%% loop through every image and get those values
for i = 1:length(allData)
    fprintf('at example %d...\n', i);
    allDataInstance = allData{i};
    result = score_capsule_shape(allDataInstance, i);
    data{i} = result;
end

%% do stuff
