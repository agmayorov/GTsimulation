function [Temp, Pres, Den, ChemComp] = Atmosphere(LLA, Model, DATE)
%   [Ne, Temp, Ni, parset] = Atmosphere([lat lon alt], Model, [year mm dd])
%                            Atmosphere([lat lon alt], Model, [year mm dd hh mn ss])
%                            Atmosphere([lat lon alt], Model, [year -ddd UTsec])
%
%   Calculate characteristics of atmosphere according to ISA or NRLMSISE-00
%   Ver. 1, red. 3 / September 2021 / A. Mayorov / CRTeam / NRNU MEPhI, Russia
%
%   Arguments:
%       lat         - Float         -   Geodetic latitude, [deg]
%       lon         - Float         -   Geodetic longitude, [deg]
%       alt         - Float         -   Geodetic altitude, [km]
%       year,mm,dd  - Int           -   Year, month, day of month
%       hh,mn,ss    - Int           -   Hour, minutes, seconds (noon if empty)
%       ddd         - Int           -   Day of year as negative value
%       UTsec       - Int           -   UT seconds
%
%   Output:
%       Temp        - Float         -   Temperature, in kelvin
%       Pres        - Float         -   Preasure, in Pa
%       Den         - Float         -   Total mass density, in g/cm3
%       ChemComp    - Float arr.    -   He, O, N2, O2, Ar, H, and N in 1/cm3
%
%   Examples:
%       [Temp, Pres, Den, ChemComp] = Atmosphere([0 90 1], 'ISA')
%       [Temp, Pres, Den, ChemComp] = Atmosphere([0 90 1], 'NRLMSISE-00', [2016 01 15]);
%       [Temp, Pres, Den, ChemComp] = Atmosphere([0 90 1], 'NRLMSISE-00', [2016 02 17 05 55 00]);
%       [Temp, Pres, Den, ChemComp] = Atmosphere([0 90 1], 'NRLMSISE-00', [2016 -15 3600]);
%       [Temp, Pres, Den, ChemComp] = Atmosphere([45 -50 10], 'NRLMSISE-00', [2007 -4 0]);
%
%   Important: warning messages are disabled
%   warning('off')

    % Initialization 
    Temp = NaN;
    Pres = NaN;
    Den = NaN;
    ChemComp =zeros(1,7);
    if nargin == 2
        DATE = [];
    end

    %{
        Very slow!
    if ~contains(path,'/lustre/mFunctions/DatesAndTime')
        AtmosphericSetup
    end    
    %}

    % Check input
    if length(LLA) ~= 3
        error('  ATMO: Wrong format - coordinates')
    end
    if length(DATE) ~= 0 && length(DATE) ~= 3 && length(DATE) ~= 6
        error('  ATMO: Wrong format - Date')
    end
    if ~ischar(Model)
        error('  ATMO: Wrong Model name - not a string')
    end    
    if ~ismember(Model,{'ISA','NRLMSISE-00'})
        error('  ATMO: Wrong atmospheric model. Possible ISA, NRLMSISE-00.')
    end

    % Read input & Date [year ddd UThour UTseconds]
    lat = LLA(1);
    lon = LLA(2);
    alt = LLA(3)*1e3;   % km -> m

    if strcmp(Model,'NRLMSISE-00')
        if DATE(2) > 0
            [DATE(1), DATE(2), ~, DATE(3)] = DateRedefine(DATE);
            if length(DATE) == 6
                DATE(4:6) = [];
            end
        else
            DATE = abs(DATE);
        end    
    end

    % Constants etc
    %RE2m = 6378137;                     % Earth radius in m
    persistent f107Daily f107Average magneticIndex dayOfYear date ndate F Fav MI

    %% International Standard Atmosphere
    % https://www.mathworks.com/help/aerotbx/ug/atmosisa.html
    % https://en.wikipedia.org/wiki/International_Standard_Atmosphere
    % Input: Geopotential altitude, m (correct for 0 (surface) m to 20,0000 m)
    % Output:  Density, kg/m^3 / Speed of sound, m/s / Temp, K / Pres, Pa    
    if strcmp(Model,'ISA')
        if alt > 20e3
            return
        end
        % Mean Sea Level app. Geoid
        gpalt = geoidheight(lat,wrapTo360(lon));
        [Temp, ~, Pres, Den] = atmosisa(alt+gpalt);
        ChemComp = '';
        
    %%  NRLMSISE-00
    % Calculates the neutral atmosphere empirical model from the surface to lower exosphere (0 m to 1,000,000 m)
    % https://www.mathworks.com/help/aerotbx/ug/atmosnrlmsise00.html
    % https://ccmc.gsfc.nasa.gov/modelweb/atmos/nrlmsise00.html
    elseif strcmp(Model,'NRLMSISE-00')
        if alt > 1000e3 || alt < 0
            return
        end
        if isempty(f107Average)
            load atmosnrlmsise00_input.mat
            ndate = [0 0 0];
        end
        if ~isequal(ndate, DATE)
            %[~,ia,~] = intersect(date(:,1:3),varargin{3}(1:3),'rows');
            ia = find(date(:,1) == DATE(1) & dayOfYear == DATE(2),1,'first');
            % f107Daily, f107Average, Magnetic Index during a day ia
            Fav = f107Average(ia);
            MI = magneticIndex(ia,:);
            F = NaN;
            while isnan(F)
                F = f107Daily(ia);
                ia = ia + 1;
            end
            ndate = DATE;
        end
        [T, nrlmsise] = atmosnrlmsise00(alt, lat, lon, DATE(1), DATE(2), DATE(3), Fav, F, MI);
        Temp = T(:,2);
        Pres = '';
        Den = nrlmsise(:,6);
        ChemComp = [nrlmsise(1:5) nrlmsise(7:8)]./1e6; % He, O, N2, O2, Ar, H, and N in 1/cm3
        %ChemComp = ChemComp./sum(ChemComp);
    end

    Den = Den*1e-3;     % kg/m3 -> g/cm3