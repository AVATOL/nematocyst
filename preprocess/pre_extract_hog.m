function [ featureMatrix ] = pre_extract_hog( image, patchSize )
%PRE_EXTRACT_HOG Extract regular grid of HOG patches from an image.
%   image:          image matrix
%   patchSize:      patch size
%   featureMatrix:  matrix of patches of features

%% argument checking
narginchk(2, 2);

%% constants
HOG_CELL_SIZE = 8;

%% setup
[M, N] = size(image);

dim = size(func_extract_hog_vector(image(1:patchSize, 1:patchSize), HOG_CELL_SIZE), 1);
featureMatrix = zeros(M/patchSize, N/patchSize, dim);

%% process patches
c = 1;
for col = 1:patchSize:N
    r = 1;
    for row = 1:patchSize:M
        ycomp = row:row+patchSize-1;
        xcomp = col:col+patchSize-1;

        %% get features
        patch = image(ycomp, xcomp);

        featureVector = func_extract_hog_vector(patch, HOG_CELL_SIZE);
        featureMatrix(r, c, :) = featureVector';
        
        r = r + 1;
    end
    
    c = c + 1;
end

end

