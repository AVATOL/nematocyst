close all;

%% store all the eccentricies and minor/major axis values
data = cell(size(allData));

%% loop through every image and get those values
for i = 1:length(allData)%[115 47 72 102 94 124]%[115 47 72]%[102 94 124]
    fprintf('at example %d...\n', i);
    allDataInstance = allData{i};
    detect_capsule_from_basal(allDataInstance, i);
end

%% do stuff
