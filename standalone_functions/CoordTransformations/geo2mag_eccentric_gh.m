function [X, Y, Z] = geo2mag_eccentric_gh(x, y, z, j, g, h)
%   Input: x, y, z - scalar - [meters], 
%   DTnum - serial date number format (datenum)
%   j >  0 GEO to MAG
%   j <= 0 MAG to GEO

    RE = 6378137.1;
    A = DisplacementForEccentricDipole_gh(g, h).*RE; % Displacement in meters

    if numel(x) > 1 && iscolumn(x)
        x = x';
        y = y';
        z = z';
    end

    if j > 0
        vec = [0.339067758413505 -0.919633920274268 -0.198258689306225 ; ...
               0.938257039240758 0.345938908356903 0 ; ...
               0.068589929661063 -0.186019809236783 0.980148994857721] * ([x; y; z] - A);
    else
        vec =([0.339067758413505 -0.919633920274268 -0.198258689306225 ; ...
               0.938257039240758 0.345938908356903 0; ...
               0.068589929661063 -0.186019809236783 0.980148994857721]' * ([x; y; z])) + A;
    end

    vec = vec';

    X = vec(:,1); 
    Y = vec(:,2);
    Z = vec(:,3);
