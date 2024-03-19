classdef BfieldAbs < handle
    properties
        ModelName
        Region
    end

    methods (Abstract)        
        [Bx, By, Bz] = GetBfield(self, x, y, z, t)
    end
end

