classdef Sumed < BfieldAbs
    
    properties
        Fields, Coefs
    end
    
    methods
        function self = Sumed(Param)
            arguments
                Param.Fields
                Param.Coefs = 0
            end
            if Param.Coefs ~= 0
                if length(Param.Coefs) ~= length(Param.Fields)
                    error("Number of coefficents must be equal to the number of fields")
                end
            else
                Param.Coefs = ones(size(Param.Fields));
            end
            self.ModelName = "Sumed";
            self.Region = "M";
            self.Fields = Param.Fields;
            self.Coefs = Param.Coefs;
        end
        
        function [Bx, By, Bz] = GetBfield(self, x, y, z, t)
            [Bx, By, Bz] = deal(0);
            for i=1:length(self.Fields)
                [Bx_f, By_f, Bz_f] = self.Fields(1).GetBfield(x, y, z, t);
                Bx = Bx + self.Coefs(i)*Bx_f;
                By = By + self.Coefs(i)*By_f;
                Bz = Bz + self.Coefs(i)*Bz_f; 
            end
        end
    end
end

