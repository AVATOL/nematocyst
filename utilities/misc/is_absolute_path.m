function [ result ] = is_absolute_path( path )
%IS_FULL_FILE Returns whether the path is absolute or otherwise relative.

% absolute path begins with / for linux systems or X: for Windows
result = (length(path) > 1 && path(1) == '/')...
    || (length(path) > 2 && path(2) == ':');

end

