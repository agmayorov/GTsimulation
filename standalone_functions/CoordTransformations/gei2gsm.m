%               FOR YEARS 1901 THROUGH 2099
%               j>0: GEI -> GSM ; j<=0: GSM -> GEI
%               ���, ����, ������� � UT
function [X,Y,Z] = gei2gsm(x,y,z,Year,Day,Secs,j)
 %                                  S
 
 [S, GST, ~] = DirectionEarthtoSun(Year,Day,Secs); 
 
 %                                  MAG Z -> GEO Z
 
 D = [0.068589929661063; -0.186019809236783; 0.980148994857721];
 
 %                                  GEO Z -> GEI Z
 
 D = [cosd(GST) -sind(GST) 0; sind(GST) cosd(GST) 0; 0 0 1]*D;
 
 %                                  GEI Z and Sun axis -> GSM Y
 
 a=cross(D,S);
 Y=a./sqrt(sum(a.^2));

 %                                  GSM Z
 
 Z=cross(S,Y);

 %                      j<=0: GEI -> GSM ; j>0: GSM -> GEI
 
 if j>0
     vec=[S; Y; Z]*[x; y; z];
 else
     vec=[S; Y; Z]'*[x; y; z];
 end
 X=vec(1, :); 
 Y=vec(2, :);
 Z=vec(3, :);
end
 
 