function [ inferImage ] = visualize_edge_weights( image, labels, label2color, segMat, wAdjMat )
%VISUALIZE_EDGE_WEIGHTS Visualize single result.
%
%   image:          image matrix
%   labels:         node labels vector
%   label2color:    mapping from label to color
%   segMat:         segments matrix
%   wAdjMat:        adjacency matrix with edge weights

narginchk(5, 5);

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
        
        %% draw pixel assuming not a boundary
%         inferImage(row, col, :) = reshape([0 0 128], [1 1 3]);
        inferImage(row, col, :) = (1-alpha)*inferImage(row, col, :) + reshape(uint8(alpha*color'), [1 1 3]);
        
        %% if boundary between segments, color as segment boundary
        boundary = 0; % 0 nothing, 1 boundary
        if col ~= 1
            prevSegmentId = segMat(row, col-1);
            if prevSegmentId ~= segmentId
                boundary = 1;
            end
        end
        
        if boundary == 1
            weight = wAdjMat(segmentId, prevSegmentId); % should be [0, 1]
            color = uint8(weight * 255);
            inferImage(row, col, :) = reshape([color color color], [1 1 3]);
        end
    end
end

for col = 1:width
    for row = 1:height
        segmentId = segMat(row, col);
        label = labels(segmentId);
        
        %% if boundary between segments, color as segment boundary
        boundary = 0; % 0 nothing, 1 boundary
        if row ~= 1
            prevSegmentId = segMat(row-1, col);
            if prevSegmentId ~= segmentId
                boundary = 1;
            end
        end
        
        if boundary == 1
            weight = wAdjMat(segmentId, prevSegmentId); % should be [0, 1]
            color = uint8(weight * 255);
            inferImage(row, col, :) = reshape([color color color], [1 1 3]);
        end
    end
end

end

