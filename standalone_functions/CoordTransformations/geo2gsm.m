function [X, Y, Z] = geo2gsm(x, y, z, Year, DoY, Secs, d)
    [S, GST, ~] = DirectionEarthtoSun(Year, DoY, Secs);
    D = [cosd(GST) -sind(GST) 0; sind(GST) cosd(GST) 0; 0 0 1]* [0.068589929661063; -0.186019809236783; 0.980148994857721];
    a = cross(D,S);
    Y = a./sqrt(sum(a.^2));
    Z = cross(S,Y);

    if numel(x) > 1 && iscolumn(x)
        x = x';
        y = y';
        z = z';
    end

    if d == 1
        vec = [S; Y; Z]*([cosd(GST) -sind(GST) 0; sind(GST) cosd(GST) 0; 0 0 1]*[x; y; z]);
    else
        vec = [cosd(GST) -sind(GST) 0; sind(GST) cosd(GST) 0; 0 0 1]'*([S; Y; Z]'*[x; y; z]);
    end

    vec = vec';

    X=vec(:, 1); 
    Y=vec(:, 2);
    Z=vec(:, 3);
