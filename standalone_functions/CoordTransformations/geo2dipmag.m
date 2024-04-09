function [X, Y, Z] = geo2dipmag(x, y, z, psi, j)
%   j >  0 GEO to DIPMAG
%   j <= 0 DIPMAG to GEO

    if j>0
        vec = [ cosd(psi) 0 sind(psi); 
                0 1 0; 
                -sind(psi) 0 cosd(psi) ] * [x; y; z];
    else
        vec = [ cosd(psi) 0 sind(psi); 
                0 1 0; 
                -sind(psi) 0 cosd(psi) ]' * [x; y; z];
    end

    X=vec(1, :); 
    Y=vec(2, :);
    Z=vec(3, :);
