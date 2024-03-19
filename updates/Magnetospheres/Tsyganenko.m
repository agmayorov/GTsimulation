classdef Tsyganenko < BfieldAbs
    properties
        ps, iopt, Year, DoY, Secs, ModCode
    end
    
    methods
        function self = Tsyganenko(TParam)
            arguments
                TParam.ps
                TParam.iopt
                TParam.Year
                TParam.DoY
                TParam.Secs
                TParam.ModCode
            end
            self.ModelName = "Tsyg";
            self.Region = "M";
            self.ps = TParam.ps;
            self.iopt = TParam.iopt;
            self.Year = TParam.Year;
            self.DoY = TParam.DoY;
            self.Secs = TParam.Secs;
            self.ModCode = TParam.ModCode;
        end
        
        function [X, Y, Z] = GEOtoGSM(self, x, y, z, d)
            [S, GST, ~] = DirectionEarthtoSun(self.Year, self.DoY, self.Secs);
            D = [cosd(GST) -sind(GST) 0; sind(GST) cosd(GST) 0; 0 0 1]* [0.06859; -0.18602; 0.98015];
            a = cross(D,S);
            Y = a./sqrt(sum(a.^2));
            Z = cross(S,Y);
            
            if d == 1
                vec = [S; Y; Z]*([cosd(GST) -sind(GST) 0; sind(GST) cosd(GST) 0; 0 0 1]*[x; y; z]);
            else
                vec = [cosd(GST) -sind(GST) 0; sind(GST) cosd(GST) 0; 0 0 1]'*([S; Y; Z]'*[x; y; z]);
            end
            
            X = vec(1); 
            Y = vec(2);
            Z = vec(3);
        end

        function [Bx, By, Bz] = GetBfield(self, X, Y, Z, ~)
            arguments
                self, X, Y, Z, ~
            end
            [X, Y, Z] = self.GEOtoGSM(X, Y, Z, 1);

            if self.ModCode == 89
                [Bx, By, Bz] = T89d(self.iopt, self.ps, X, Y, Z);
            elseif self.ModCode == 96
                [Bx, By, Bz] = T96(self.iopt, self.ps, X, Y, Z);
            end
        
            [Bx, By, Bz] = self.GEOtoGSM(Bx, By, Bz, 0);
        end
    end

end



