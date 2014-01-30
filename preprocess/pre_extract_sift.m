function [ featureMatrix ] = pre_extract_sift( image, patchSize )
%PRE_EXTRACT_SIFT Extract regular grid of SIFT patches from an image.
%   image:          image matrix
%   patchSize:      patch size
%   featureMatrix:  matrix of patches of features

%% argument checking
narginchk(2, 2);

%% setup
[M, N] = size(image);

SIFTDIM = 128;
featureMatrix = zeros(M/patchSize, N/patchSize, SIFTDIM);

%% process patches
im = im2single(image);
delta = patchSize;
xr = delta/2:delta:delta/2+delta*(N/patchSize-1);
yr = delta/2:delta:delta/2+delta*(M/patchSize-1);
[x,y] = meshgrid(xr,yr);
fr = [x(:)'; y(:)'];
fr(end+1,:) = delta/2;

[frames, patches] = vl_covdet(im, 'frames', fr, 'estimateAffineShape', true, 'estimateOrientation', true);

%% assign patches
for ind = 1:size(frames, 2)
    col = (frames(1, ind)-patchSize/2)/patchSize+1;
    row = (frames(2, ind)-patchSize/2)/patchSize+1;
    
    if ~any(featureMatrix(row, col, :))
        featureMatrix(row, col, :) = patches(:, ind);
    end
end

end

