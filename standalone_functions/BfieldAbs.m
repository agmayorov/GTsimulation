classdef BfieldAbs < handle
    properties
        ModelName
        Region
        UseMeters
        UseTeslas
    end

    methods (Abstract)        
        [Bx, By, Bz] = GetBfield(self, x, y, z, t)
    end
end

