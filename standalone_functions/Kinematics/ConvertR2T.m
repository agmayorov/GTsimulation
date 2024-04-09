function E=ConvertR2T(R,M,A,Z)
%% Converts rigidity into Energy
%% INPUT:
%%       R - rigidity in GV
%%       M - Mass of particle in GV/c^2
%%       A - Number of nuclons
%%       Z - Charge
%% OUTPUT:
%%       E - Kinetic energy in GeV
    A(A==0) = 1;
    E = (1./A).*(sqrt(power((Z.*R),2)+power(M,2))-M);