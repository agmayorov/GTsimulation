function [Year, DoY, Secs, DTnum] = YDS(Date)

    Year  = Date(1);
    DoY   = day(datetime(Date(1:3)), 'dayofyear');
    if numel(Date) == 6
        Secs  = Date(4)*3600 + Date(5)*60 + Date(6);
    else
        Secs = 43200;
    end
    DTnum = datenum(Date(1:3));