function [ outputPath ] = normalize_file_sep( outputPath )
%NORMALIZE_FILE_SEP Replace file separator in string with system's

outputPath = strrep(outputPath, '/', filesep);
outputPath = strrep(outputPath, '\', filesep);

end

