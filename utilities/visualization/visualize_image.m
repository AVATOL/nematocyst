function [ inferImage ] = visualize_image( image, labels, label2color, segMat, cutMat )
%VISUALIZE_IMAGE Visualize single result.
%
%   image:          image matrix
%   labels:         node labels vector
%   label2color:    mapping from label to color
%   segMat:         segments matrix
%   cutMat:         (optional) stochastic cuts matrix

narginchk(4, 5);

cutAvailable = 1;
if nargin < 5
    cutAvailable = 0;
end

%% alpha parameter controls blending of label color and original image
alpha = 0.66;

%% initialization
inferImage = im2uint8(image);
[height, width, nChannels] = size(image);

if nChannels == 1
    inferImage = uint8(zeros(height, width, 3));
    inferImage(1:height, 1:width, 1) = im2uint8(image);
    inferImage(1:height, 1:width, 2) = im2uint8(image);
    inferImage(1:height, 1:width, 3) = im2uint8(image);
end

%% create new visualization image
for row = 1:height
    for col = 1:width
        segmentId = segMat(row, col);
        label = labels(segmentId);
        color = label2color(label);
        
        %% draw pixel color based on label
        inferImage(row, col, :) = (1-alpha)*inferImage(row, col, :) + reshape(uint8(alpha*color'), [1 1 3]);
        
        %% if boundary between segments, color as segment boundary
        boundary = -1; % 0 nothing, 1 no boundary, 2 cut boundary
        if col ~= 1
            prevSegmentId = segMat(row, col-1);
            if prevSegmentId ~= segmentId
                if ~cutAvailable
                    boundary = 0;
                elseif cutMat(prevSegmentId, segmentId) ~= 0
                    boundary = 1;
                else
                    boundary = 2;
                end
            end
        end
        
        %% color label, boundary or stochastic cut
        if boundary > 0
            if boundary == 0 || boundary == 1
                color = uint8(0);
            elseif boundary == 2
                color = uint8(255);
            end
            inferImage(row, col, :) = reshape([color color color], [1 1 3]);
        end
    end
end

for col = 1:width
    for row = 1:height
        segmentId = segMat(row, col);
        label = labels(segmentId);
        color = label2color(label);
        
        %% if boundary between segments, color as segment boundary
        boundary = -1; % 0 nothing, 1 no boundary, 2 cut boundary
        if row ~= 1
            prevSegmentId = segMat(row-1, col);
            if prevSegmentId ~= segmentId
                if ~cutAvailable
                    boundary = 0;
                elseif cutMat(prevSegmentId, segmentId) ~= 0
                    boundary = 1;
                else
                    boundary = 2;
                end
            end
        end
        
        %% color label, boundary or stochastic cut
        if boundary > 0
            if boundary == 0 || boundary == 1
                color = uint8(0);
            elseif boundary == 2
                color = uint8(255);
            end
            inferImage(row, col, :) = reshape([color color color], [1 1 3]);
        end
    end
end

end

