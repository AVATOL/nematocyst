function [ objects ] = annotate_image_polygon( img )
%ANNOTATE_IMAGE_POLYGON Annotate an image with polygons
%
%   img:        image loaded
%   objects:    cell of structs where a struct is an annotation instance
%               and each instance struct has
%                   .xcoords:   list of x coordinates
%                   .ycoords:   list of y coordinates
%                   .charID:    character ID
%                   .charState: character state

objects = cell(1, 1);
cnt = 1;

done = 0;
% loop for multiple annotations in one image
while ~done
    good = 0;
    
    % loop for annotation attempt
    while ~good
        figure('name', 'Polygon Annotation Tool');
        imshow(img);
        title(sprintf('Annotation Object %d', cnt));
        
        % store coordinates for this object
        obj.xcoords = [];
        obj.ycoords = [];
        
        % record coordinates and draw them
        button = 0;
        xprev = [];
        yprev = [];
        while button ~= 3 % while not a right click
            % get input
            [x,y,button] = ginput(1);
            
            % if not a right click
            if button ~= 3
                % draw markers
                hold on;
                
                if ~isempty(xprev)
                    plot([xprev x], [yprev y], '-y+', 'LineWidth', 2);
                else
                    plot(x,y,'y+');
                end
                
                % add to list
                obj.xcoords = [obj.xcoords; x];
                obj.ycoords = [obj.ycoords; y];
                
                % update
                xprev = x;
                yprev = y;
            else
                % connect first and last point (visually only)
                if length(obj.xcoords) >= 1
                    plot([obj.xcoords(1) obj.xcoords(end)], ...
                            [obj.ycoords(1) obj.ycoords(end)], '-y+', 'LineWidth', 2);
                end
            end
        end
        
        good = strcmp(questdlg('Accept annotation?'), 'Yes') == 1;
        if ~good
           close; 
        end
    end
    
    % get character ID/state of annotation
    temp = inputdlg('Enter character ID:');
    obj.charID = temp{1};
    temp = inputdlg('Enter character state:');
    obj.charState = temp{1};
    
    % add to objects data structure
    objects{cnt} = obj;
    cnt = cnt + 1;
    
    % add another?
    done = strcmp(questdlg('Add another annotation for this image?'), 'No') == 1;
    
    close;
end

end

