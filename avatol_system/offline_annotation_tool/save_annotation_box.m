function save_annotation_box(objects, filename)
% saves annotation objects to file
% format: x1,y1;...;xn,yn:charID:charState

fid = fopen(filename, 'w');

for i = 1:length(objects)
   obj = objects{i};
   
   string = sprintf('%.f,%.f', obj.x, obj.y); 
   string = sprintf('%s:%s', string, obj.windowSize);
   
   fprintf(fid, '%s\n', string);
end

fclose(fid);

end