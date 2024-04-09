function R=ConvertT2R(T,M,A,Z)
%% Converts particles energy into  rigidity
%% INPUT:
%%       T - Kinetic energy in GeV
%%       M - Mass of particle in GV/c^2
%%       A - Number of nuclons
%%       Z - Charge
%% OUTPUT:
%%       R - Rigidity in GV

    A(A==0) = 1;
    R = (1./Z).*(sqrt(power((A.*T+M),2)-power(M,2)));
