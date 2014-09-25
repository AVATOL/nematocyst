function [ outputPath ] = normalize_file_sep( outputPath, fileSeparator )
%NORMALIZE_FILE_SEP Replace file separator in string with system's
%   if second argument is included, force replacement with that separator

if nargin < 2
    outputPath = fullfile(outputPath);
else
    if fileSeparator ~= '/' && fileSeparator ~= '\'
        error('second argument is not a valid file separator. needs to be / or \');
    end
    
    outputPath = strrep(outputPath, '/', fileSeparator);
    outputPath = strrep(outputPath, '\', fileSeparator); 
end

end
