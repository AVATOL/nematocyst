function [ inferImage ] = visualize_grid_image( image, labels, label2color, segMat, cutMat )
%VISUALIZE_IMAGE Visualize single result.
%
%   image:          image matrix
%   labels:         node labels vector
%   label2color:    mapping from label to color
%   segMat:         segments matrix
%   cutMat:         (optional) stochastic cuts matrix

narginchk(4, 5);

%% patch size
patchSize = 32;

%% setup
clf
imshow(image);
hold on

[M, N] = size(image);

inferenceMatrix = reshape(labels, [N/patchSize, M/patchSize]);
inferenceMatrix = inferenceMatrix';

%% draw regular grid
for k = 1:patchSize:M
    x = [1 N];
    y = [k k];
    plot(x,y,'Color','w','LineStyle','-');
    plot(x,y,'Color','k','LineStyle',':');
end

for k = 1:patchSize:N
    x = [k k];
    y = [1 M];
    plot(x,y,'Color','w','LineStyle','-');
    plot(x,y,'Color','k','LineStyle',':');
end

%% highlight positive examples
c = 1;
for col = 1:patchSize:N
    r = 1;
    for row = 1:patchSize:M
        ycomp = row:row+patchSize-1;
        xcomp = col:col+patchSize-1;
        
        inferLabel = inferenceMatrix(r, c);
%         truthLabel = truthMatrix(r, c);
        if inferLabel > 0
            visualLabel = label2color(inferLabel)./255;
            rectangle('Position', [xcomp(1), ycomp(1), patchSize, patchSize],...
                'EdgeColor', visualLabel, 'LineStyle', '-', 'LineWidth', 2);
%         if inferLabel > 0 && truthLabel > 0
%             rectangle('Position', [xcomp(1), ycomp(1), patchSize, patchSize],...
%                 'EdgeColor', visualLabel, 'LineStyle', '-', 'LineWidth', 4);
%         elseif inferLabel > 0 && truthLabel <= 0
%             rectangle('Position', [xcomp(1), ycomp(1), patchSize, patchSize],...
%                 'EdgeColor', visualLabel, 'LineStyle', '-', 'LineWidth', 2);
%         elseif inferLabel <= 0 && truthLabel > 0
%             rectangle('Position', [xcomp(1), ycomp(1), patchSize, patchSize],...
% %                 'EdgeColor', 'w', 'LineStyle', '-', 'LineWidth', 4);
%         elseif inferLabel == -1
%             visualLabel = 'k';
%             rectangle('Position', [xcomp(1), ycomp(1), patchSize, patchSize],...
%                 'EdgeColor', visualLabel, 'LineStyle', '-', 'LineWidth', 2);
%         elseif inferLabel == 0
%             visualLabel = 'w';
%             rectangle('Position', [xcomp(1), ycomp(1), patchSize, patchSize],...
%                 'EdgeColor', visualLabel, 'LineStyle', '-', 'LineWidth', 2);
        end
        
        r = r + 1;
    end
    
    
    c = c + 1;
end

hold off

end

