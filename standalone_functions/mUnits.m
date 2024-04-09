function Units = mUnits
%   Ver. 1, red. 1 / 31 March 2023 / A. Mayorov

    Units.MeV2kg = 1.7826619216224e-30;           % MeV/c2 to kg conversion
    Units.MeV2g  = 1.7826619216224e-27;           % MeV/c2 to  g conversion
    
    Units.km2m  = 1e3;                            % km to m conversion
    Units.AU2m  = 149.597870700e9;                % Astronomical unit in m
    Units.pc2m  = 3.08567758149e16;               % Parsec in m
    Units.kpc2m = 3.08567758149e19;               % kParsec in m
    
    Units.fm2cm = 1e-13;                          % femto-m in sm

    Units.RE2m  = 6378137.1;                      % Earth radius in m
    Units.RE2km = 6378.1371;                      % Earth radius in km
    Units.RM2m = 1737400;                         % Moon radius in km
    Units.RM2km = 1737.4;                         % Moon radius in m