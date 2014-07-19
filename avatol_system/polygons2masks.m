function outMask = polygons2masks( img, objects )
%POLYGONS2MASKS Converts polygon objects into an image mask.
%
%   img:        image loaded
%   objects:    cell of structs where a struct is an annotation instance
%               and each instance struct has
%                   .xcoords:   list of x coordinates
%                   .ycoords:   list of y coordinates
%                   .charID:    character ID
%                   .charState: character state
%   outMask:    image mask created from polygons

%% argument checking
narginchk(2, 2);

%% constants
WHITE = 255;

[height, width, ~] = size(img);
[X, Y] = meshgrid(1:width, 1:height);

%% for each polygon, get pixels inside of polygon
outMask = zeros(height, width);
for i = 1:length(objects)
    if length(objects{i}.xcoords) < 3
        error('Object %d must contain at least 3 points for a polygon!', i);
    end
    
    mask = inpolygon(X, Y, objects{i}.xcoords, objects{i}.ycoords);
    outMask = outMask | mask;
end

%% final output mask
outMask = WHITE*uint8(outMask);

end
