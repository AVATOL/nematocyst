function writelog( fid, msg )
%WRITELOG Write to log file and print out message.
%
%   fid :   file handler to log file   
%   msg :   message to print

narginchk(2, 2);

%% print to console
fprintf(msg);

%% write to file
fprintf(fid, sprintf('[%s] %s', datestr(now), msg));

end

