function save_annotation_polygon(objects, filename)
% saves annotation objects to file
% format: x1,y1;...;xn,yn:charID:charState

fid = fopen(filename, 'w');

for i = 1:length(objects)
   obj = objects{i};
   
   string = sprintf('%.f,%.f', obj.xcoords(1), obj.ycoords(1)); 
   for j = 2:length(obj.xcoords)
      string = strcat(string, sprintf(';%.f,%.f', obj.xcoords(j), obj.ycoords(j))); 
   end
   string = sprintf('%s:%s:%s', string, obj.charID, obj.charState);
   
   fprintf(fid, '%s\n', string);
end

fclose(fid);

end