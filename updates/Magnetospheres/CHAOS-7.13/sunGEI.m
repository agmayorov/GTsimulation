function [S,GST] = sunGEI(md2000)

% function [S,GST] = sunGEI(md2000)
% returns the unit vector position of the sun in GEI coordinates
% it also returns grenwich mean sidereal time (GST)in radians
% md2000 is modified Julian day 2000
% alternate calling: S = sunGEI(md2000)
% valid between years 1901 to 2099

rad = 180/pi;

dj = md2000+36524.5; % corrected =365*(2000-1900)+mod(2000-1901,1)+doy-0.5
% dj = md2000+36159; % old, incorrected value
fday = mod(md2000, 1);


t = dj/36525;
vl = mod(279.696678 + 0.9856473354*dj, 360);
GST = mod(279.690983 + 0.9856473354*dj + 360*fday + 180, 360)/rad; % corrected
% GST = mod(279.696678 + 0.9856473354*dj+360*fday+180, 360)/rad; % old, incorrected value
g = mod(358.475845 + 0.985600267*dj, 360.0)/rad;
slong = vl+(1.91946-0.004789*t).*sin(g) + 0.020094*sin(2*g);
obliq = (23.45229-0.0130125*t)/rad;
slp = (slong-0.005686)/rad;
sind = sin(obliq).*sin(slp);
cosd = sqrt(1-sind.^2);
SDEC = rad*atan(sind./cosd);
SRASN = 180-rad*atan2(1./tan(obliq).*sind./cosd, -cos(slp)./cosd);

X = cos(SRASN/rad).*cos(SDEC/rad);
Y = sin(SRASN/rad).*cos(SDEC/rad);
Z = sin(SDEC/rad);
S = [X Y Z];

