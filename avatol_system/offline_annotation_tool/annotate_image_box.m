function [ objects ] = annotate_image_box( img, windowSize )
%ANNOTATE_IMAGE_BOX Annotate an image with fixed bounding boxes
%
%   img:        image loaded
%   windowSize: window size (should be divisible by 8)
%   objects:    cell of structs where a struct is an annotation instance
%               and each instance struct has
%                   .x:   top left x coordinate of box
%                   .y:   top left y coordinate of box
%                   .windowSize:    window size

%% argument checking
narginchk(2, 2);

%% initialize
objects = cell(0, 0);
cnt = 1;

%% loop for multiple annotations in one image
done = 0;
while ~done
    good = 0;
    
    %% loop for annotation attempt
    while ~good
        figure('name', 'Polygon Annotation Tool');
        imshow(img);
        title(sprintf('Annotation Object %d', cnt));
        hold on;
        
        %% draw previous annotations
        for i = 1:length(objects)
            drawWindow(objects{i}.x, objects{i}.y, objects{i}.windowSize, '-w');
        end
        
        %% get input
        [x,y,~] = ginput(1);

        %% draw markers
        drawWindow(x, y, windowSize, '-y');

        %% add to list
        obj.x = x;
        obj.y = y;
        obj.windowSize = windowSize;

        good = strcmp(questdlg('Accept annotation?'), 'Yes') == 1;
        if ~good
           close; 
        end
    end
    
    %% add to objects data structure
    objects{cnt} = obj;
    cnt = cnt + 1;
    
    %% add another?
    done = strcmp(questdlg('Add another annotation for this image?'), 'No') == 1;
    
    close;
end

end

function drawWindow(x, y, windowSize, style)

plot([x x+windowSize], [y y], style, 'LineWidth', 2);
plot([x x], [y y+windowSize], style, 'LineWidth', 2);
plot([x x+windowSize], [y+windowSize y+windowSize], style, 'LineWidth', 2);
plot([x+windowSize x+windowSize], [y y+windowSize], style, 'LineWidth', 2);

end