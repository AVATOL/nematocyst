function [ nodeLocations, nodeSizes ] = pre_extract_node_locations( segments, nSegments, NORMALIZE_LOCATIONS )
%PRE_EXTRACT_NODE_LOCATIONS Summary of this function goes here
%   Detailed explanation goes here

%% argument checking
narginchk(2, 3);

if nargin < 3
    NORMALIZE_LOCATIONS = 1;
end

[imgHeight, imgWidth] = size(segments);

%% setup
nodeLocations = zeros(nSegments, 2);
nodeSizes = zeros(nSegments, 1);

%% find centroid for each segment
for i = 1:nSegments
    %% get segment
    segment = segments;
    segment(segment ~= i) = 0;
    segment(segment == i) = 1;
    
    nPixels = sum(sum(segment));
    
    xMat = repmat(1:imgWidth, imgHeight, 1);
    yMat = repmat((1:imgHeight)', 1, imgWidth);
    
    xMat = segment .* xMat;
    yMat = segment .* yMat;
    
    %% compute centroid
    x = sum(sum(xMat))/nPixels;
    y = sum(sum(yMat))/nPixels;
    
    if NORMALIZE_LOCATIONS
        x = x/imgWidth;
        y = y/imgHeight;
    else
        x = floor(x);
        y = floor(y);
    end
    
    nodeLocations(i, 1) = x;
    nodeLocations(i, 2) = y;
    
    nodeSizes(i) = nPixels;
end

end

