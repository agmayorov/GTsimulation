%input год, день, секунда в UT; 
%output единичный вектор Земля->Солнце, GST, SLONG
% PROGRAM TO CALCULATE SIDEREAL TIME AND POSITION OF THE SUN. 
% GOOD FOR YEARS 1901 THROUGH 2099. ACCURACY 0.006 DEGREE.
% INPUT IS Year, Day (INTEGERS), AND Secs, DEFINING UN. TIME. 
% OUTPUT IS GREENWICH MEAN SIDEREAL TIME (GST) IN DEGREES,
% LONGITUDE ALONG ECLIPTIC (SLONG), AND APPARENT RIGHT ASCENSION
% AND DECLINATION (SRASN, SDEC) OF THE SUN, ALL IN DEGREES
function [Vector, GST, SLONG] = DirectionEarthtoSun(Year,Day,Secs)
RAD=57.2958;
 if (Year>1901 || Year<2099) 
     FDAY = Secs/86400;
     DJ = 365* (Year-1900) + (Year-1901)/4 + Day + FDAY -0.5;
     T = DJ / 36525; 
     VL = mod (279.696678 + 0.9856473354*DJ, 360);
     GST = mod (279.690983 + 0.9856473354*DJ + 360*FDAY + 180, 360);
     G = mod (358.475845 + 0.985600267*DJ, 360) / RAD; 
     SLONG = VL + (1.91946 -0.004789*T)*sin(G) + 0.020094*sin (2*G); 
     OBLIQ = (23.45229 -0.0130125*T) / RAD; 
     SLP = (SLONG -0.005686) / RAD; 
     sinDD = sin (OBLIQ)*sin (SLP); 
     cosDD = sqrt(1-sinDD^2);
     SDEC = RAD * atan (sinDD/cosDD); 
     SRASN = 180 -RAD*atan2 (cot (OBLIQ)*sinDD/cosDD, -cos (SLP)/cosDD); 
 end
 Ex = cosd(SRASN) *cosd(SDEC);
 Ey = sind(SRASN) *cosd(SDEC);
 Ez = sind(SDEC);
 Vector = [Ex, Ey, Ez]; 
end