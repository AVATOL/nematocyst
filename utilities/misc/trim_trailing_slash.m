function [ outputPath ] = trim_trailing_slash( outputPath )
%TRIM_TRAILING_SLASH Remove trailing slash from the directory if need be

keepTrimming = 1;
while keepTrimming
    if isequal(outputPath(end), '/')
        outputPath = outputPath(1:end-1);
    elseif isequal(outputPath(end), '\')
        outputPath = outputPath(1:end-1);
    else
        keepTrimming = 0;
    end
end

end

