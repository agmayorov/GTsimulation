classdef Dipole < BfieldAbs    
    properties
        Property1
    end
    
    methods
        function obj = Dipole(inputArg1,inputArg2)
            %DIPOLE Construct an instance of this class
            %   Detailed explanation goes here
            obj.Property1 = inputArg1 + inputArg2;
        end
        
        function [Bx, By, Bz] = GetBfield(self, X, Y, Z, ~)
            if self.UseMeters

            end
            [Bx, By, Bz] = deal(0);
            if self.UseTeslas
                Bx = Bx.*1e-9;
                By = By.*1e-9;
                Bz = Bz.*1e-9;
            end
        end
    end
end

