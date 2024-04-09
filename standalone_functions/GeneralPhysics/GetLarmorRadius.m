function LR = GetLarmorRadius(T, B, Z, M, pitch)
%   Ver. 1, red. 1 / 31 March 2023 / A. Mayorov
    arguments
        T       (1,1)     double    % Kinetic energy, MeV
        B       (1,1)     double    % Magnetic field intensity, T
        Z       (1,1)     double    % Charge of particle, in e units
        M       (1,1)     double    % Mass of particle, in MeV
        pitch   (1,1)     double    % Pitch angle, deg
    end
    % Output: Larmor radius, m
    % Tested with
    %   https://www.geomagsphere.org/codes/larmor.php
    %   https://www.sciencedirect.com/topics/earth-and-planetary-sciences/larmor-radius

    Units = mUnits;
    Const = mConstants;

    gamma = (T + M) / M;
    omega = abs(Z) * Const.e * B / (gamma * M * Units.MeV2kg);
    
    LR =  sind(pitch) * sqrt(1e0-1e0/gamma^2) * Const.c / omega;
end