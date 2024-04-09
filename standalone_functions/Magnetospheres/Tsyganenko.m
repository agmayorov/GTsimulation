classdef Tsyganenko < BfieldAbs
    properties
        ps, iopt, Year, DoY, Secs, DTnum, Date, ModCode
    end
    
    methods
        function self = Tsyganenko(TParam)
            arguments
                TParam.Date
                TParam.ModCode
                TParam.UseMeters = 0
                TParam.UseTeslas = 0
            end
            self.ModelName = "Tsyg";
            self.Region = "M";
            self.UseMeters = TParam.UseMeters;
            self.UseTeslas = TParam.UseTeslas;
            self.Date = TParam.Date;
            self.ModCode = TParam.ModCode;

            [self.Year, self.DoY, self.Secs, self.DTnum] = deal(0);
            if numel(self.Date) ~= 1
                [self.Year, self.DoY, self.Secs, self.DTnum] = YDS(self.Date);
            end

            if self.Date == 0  % quiet
                self.ps = 0;
                self.iopt = [1.1200 2.0000 1.6000 -0.2000];
            else
                self.ps = self.GetPsi();
                self.iopt = self.GetTsyganenkoInd(0);
            end   

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

        function psi = GetPsi(self)
            %   Get psi angle between [0 0 1] in MAG and [0 0 1] in GSM
            %   psi - scalar - rad
            %   Data - vector - [Year Day Secs], Day - count of days in year
            %   Example: psi = GetPsi([2000, 50, 12*60*60]);
            Data = [self.Year, self.DoY, self.Secs, self.DTnum];
            [x, y, z] = geo2mag_eccentric(0, 0, 1, 0, Data(4)); 
            [x, y, z] = gei2geo(x, y, z, Data(1), Data(2), Data(3), 0);
            [x, y, z] = gei2gsm(x, y, z, Data(1), Data(2), Data(3), 1);
            psi = acos((dot([0 0 1], [x y z]))/norm([x y z]));
        end

        function ind = GetTsyganenkoInd(self, p)
            persistent T96input date ndate
            if p == 0
                load T96input_short.mat T96input date
                ndate = [0 0 0];
            end
        
            if ~isequal(ndate, self.Date)
                ia = find(date(:,1) == self.Date(1) & date(:,2) == self.Date(2) & ...
                    date(:,3) == self.Date(3) & date(:,4) == self.Date(4), 1, 'first');
                ndate = self.Date;
            end
        
            if self.ModCode == 89
                ind = T96input(ia, 5);
            elseif self.ModCode == 96
                ind = T96input(ia, 1:4);
            end
        end

        function [Bx, By, Bz] = GetBfield(self, X, Y, Z, ~)
            arguments
                self, X, Y, Z, ~
            end
            if self.UseMeters
                X = X./6378137.1;
                Y = Y./6378137.1;
                Z = Z./6378137.1;
            end

            [X, Y, Z] = self.GEOtoGSM(X, Y, Z, 1);

            if self.ModCode == 89
                [Bx, By, Bz] = T89d(self.iopt, self.ps, X, Y, Z);
            elseif self.ModCode == 96
                [Bx, By, Bz] = T96(self.iopt, self.ps, X, Y, Z);
            end
        
            [Bx, By, Bz] = self.GEOtoGSM(Bx, By, Bz, 0);

            if self.UseTeslas
                Bx = Bx.*1e-9;
                By = By.*1e-9;
                Bz = Bz.*1e-9;
            end
        end
    end

end



