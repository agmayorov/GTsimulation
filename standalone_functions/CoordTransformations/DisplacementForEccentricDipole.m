function A = DisplacementForEccentricDipole(DTnum)
%   Input:  Data - datenum format
%   Output: displacement between geocentric and eccentric dipole centers in RE

    persistent g h
    if isempty(g)
        [g, h] = loadigrfcoefs(DTnum);
    end

    B0 = norm([g(1,1) g(1,2) h(1,2)]);
    L0 = 2*g(1,1)*g(2,1) + sqrt(3)*(g(1,2)*g(2,2) + h(1,2)*h(2,2));
    L1 = -g(1,2)*g(2,1) + sqrt(3)*(g(1,1)*g(2,2) + g(1,2)*g(2,3) + h(1,2)*h(2,3));
    L2 = -h(1,2)*g(2,1) + sqrt(3)*(g(1,1)*h(2,2) - h(1,2)*g(2,3) + g(1,2)*h(2,3));
    E = (L0*g(1,1) + L1*g(1,2) + L2*h(1,2))/(4*B0^2);
    DX = (L1 - g(1,2)*E)/(3*B0^2);
    DY = (L2 - h(1,2)*E)/(3*B0^2);
    DZ = (L0 - g(1,1)*E)/(3*B0^2);
    A = [DX; DY; DZ];
