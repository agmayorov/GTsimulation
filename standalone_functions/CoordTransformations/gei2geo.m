%               FOR YEARS 1901 THROUGH 2099
%               ���, ����, ������� � UT
function [X,Y,Z] = gei2geo(x,y,z,Year,Day,Secs,j)
                            %j>0 GEI to GEO, j<=0 GEO to GEI
[~, GST, ~] = DirectionEarthtoSun(Year,Day,Secs);
if j>0
    vec=[cosd(GST) -sind(GST) 0; sind(GST) cosd(GST) 0; 0 0 1]'*[x;y;z];
else
    vec=[cosd(GST) -sind(GST) 0; sind(GST) cosd(GST) 0; 0 0 1]*[x;y;z];
end
X=vec(1, :); 
Y=vec(1, :);
Z=vec(1, :);
end
