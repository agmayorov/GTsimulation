function [N, ND] = GetNDaysInMonth(year,month)

    ND = [31 28 31 30 31 30 31 31 30 31 30 31];
    if ~mod(year,4)
        ND(2) = 29;
    end

    N = ND(month);