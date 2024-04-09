function [year, ddd, UThour, UTseconds] = DateRedefine(Date)
%   [year, ddd, UThour, UTseconds] = DateRedefine(Date)
%   Convert [year mm dd hh mm ss] to [year ddd UThour UTseconds]
%   Ver. 1, red. 1 / August 2021 / A. Mayorov / CRTeam / NRNU MEPhI, Russia
%
%   Arguments:
%       Date        - Int. array -   Date and time UT, [YYYY MM DD HH MM SS] or [YYYY MM DD] 
%                                    ([HH MM SS] corresponds to the noon)
%
%   Output:  
%       year        - int        -   Year
%       ddd         - int        -   Day of year
%       UThour      - int        -   UT hour of day
%       UTseconds   - int        -   UT seconds of day
%
%   Examples:
%       [year, ddd, UThour, UTseconds] = DateRedefine([2006 2 25])
%       [year, ddd, UThour, UTseconds] = DateRedefine([2006 2 25 01 05 00])

        if length(Date) ~= 3 && length(Date) ~= 6
            error('  DateRedef: Please set a date [year day month] or [year day month hh mm ss]')
        end
        if Date(2) < 1 || Date(2) > 12 || Date(3) < 1 || Date(3) > 31
            error('  DateRedef: Wrong MM or DD')
        end
        if length(Date) == 6 && ...
            (Date(4) < 0 || Date(4) > 23 || Date(5) < 0 || Date(5) > 59 || Date(6) < 0 || Date(6) > 59)
            error('  DateRedef: Wrong HH or MM or SS')
        end

        year = Date(1);
        ddd = datenum([Date(1),Date(2),Date(3)]) - datenum([Date(1),1,1]) + 1;
        if length(Date) == 6
            UThour = Date(4);
            UTseconds = Date(4)*3600 + Date(5)*60 + Date(6);
        else
            UThour = 12;
            UTseconds = 43200;
        end
