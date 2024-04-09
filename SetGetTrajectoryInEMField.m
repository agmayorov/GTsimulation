clear
GTaddpath

fQueue = '';

% >M<agnetosphere, >H<eliosphere, >G<alaxy
Region = 'M';
if strcmp(Region,'M')
    UsePAML3 = 0;
    if UsePAML3 == 1
        Nevents = 1:5:10;
        PAML3 = GetPAML3data(212, Nevents); 
        Ro = {'LLA', [PAML3.lat PAML3.l on PAML3.alt], 'Radius', 0, 'Units', 'ddkm'};
        Vo = {'Directions', PAML3.Sij, 'Backtracing', 'on'};
        Date = PAML3.Date;
        Particle = {'Type', 'pr', 'Rig', PAML3.Rig, 'Units', 'GV'}; % Particle type
    else
        Nevents = 1;
        Ro = {'Center', [0 0 0], 'Radius', 10, 'Units', 'RE', 'CRFotp', 'geo'};
        % {'Center', [1.5 0 0], 'Radius', 10, 'Units', 'RE', 'CRFinp', 'mag', 'CRFotp', 'geo'}
        Vo = {'Directions', [0 0 0], 'IO', 'In'};
        % {'Directions', [-1 0 0]} / {'Directions', [0 0 0], 'IO', 'In'} / 'Pitch', 45
        % 'Backtracing' : 'on'/'off'
        Particle = {'Type', 'pr', 'T', 1000, 'Units', 'MeV', 'Sbas', 'T', 'Sind', -2.7};
        % {'Type', 'pr', 'T', 30, 'Units', 'GeV'}
        % {'Type', 'pr', 'Tmin', 30, 'Tmax', 100, 'Units', 'GeV', 'Sbas', 'T', 'Sind', -2.7}
        % {'Type', '', 'Tmin', 30, 'Tmax', 100, 'Units', 'GeV', 'Sbas', 'F', 'Sind', 0.5}
        Date = [2006 07 05 12 00 00]; % UT
    end
    EMFF = {'Bfield', ...
        Gauss('Model', 'IGRF', 'Ver', 13, 'Type', 'core', 'Date', [2006 06 15], "UseMeters", 1, "UseTeslas", 1)};
%     EMFF = {"Bfield", Tsyganenko("Date", [2008 1 1 12 0 0], "ModCode", 96, "UseMeters", 1, "UseTeslas", 1);
    % 'Null' by default 
    % 'Dip' / {'Dip'} / {'Dip', 0} / {'Dip', [0 30100]} 
    % 'IGRF' / {'IGRF'}
    % 'CHAOS' / {'CHAOS'} / {'CHAOS', [20 110 0]} / {'CHAOS', [20 0 0 7.13]}
    % 'Tsyg' / {'Tsyg'} / {'Tsyg', 96} / {'Tsyg', [89 pi/4 5]} / {'Tsyg', [96 0 2 0 0 0]} 
    % {'IGRF', 'Tsyg', 'CHAOS', [20 110 0]}
    Steps = {'GTalgo', 'BB', 'NLarmor', 100, 'Nsteps', 1e3};
    % {'TimeStep', 0.1e-4, 'Nsteps', 10} / {'GTAlgo', 'RK4', 'TimeStep', 0.01, 'Time', 1}
    Medium = ''; %{'Atmo', 'NRLMSISE-00'}; %{'Atmo', 'NRLMSISE-00', 'Iono', 'IRI-2016'};
    % '' / {'Atmo', 'NRLMSISE-00'} / {'Iono', 'IRI-2016'}
    InteractEM = {'RadLoss', 'off'};
    % {'RadLoss', 'on'/'off'}
    InteractNUC = ''; %{'IntPathDen', 5};
    % IntPathDen, EAS, GenMax, EnMin, 0.01, Exclude, [12 14 16] / 'l'
    %{'IntPathDen', 5, 'GenMax', 1 'EnMin', 0.1, 'Exclude', 'l'};
    SaveMode = {'Coordinates', 'on', 'Energy', 'on'};
	% 'UserSaveGTCondition', @MyFuncSaveGTCondition
    % Coordinates Clocks Bfield Efield Energy Angles TrackInfo
    BC = {'Rmin', 1, 'Rmax', 6, 'Units', 'RE', 'MaxRev', 1};
    % {'Rmin', 1, 'Rmax', 30, 'Units', 'RE', 'MaxRev', 1 'UserBCFunction', @testBC}
    % 'UserBCFunction', @BCFullRevolution
    GTmag = '';
    % GTmag = {'TrackParam', 'on', 'ParticleOrigin', 'on'};
    % 'TrackParam', 'ParticleOrigin' : 'on'/'off'
    IOinfo  = ''; %{'SaveDir', 'test', 'SaveFile', 'test'};
    % Nfiles, SaveDir, SaveFile, LoadGTtrack 0/1
    % {'Nfiles', 2, 'SaveFile', 'test'}
    Verbose = 2;
    % 0 - no output info / 1 - short output 
    % 2 - long output / 3 - with secondaries

elseif strcmp(Region,'H') % Need to be updated
    Ro = {'Center', [0 0 0], 'Radius', 50, 'Units', 'AU'};
    Vo = {'Directions', [0 0 0], 'IO', 'In'};
    Particle = {'Type', 'pr', 'T', 1, 'Units', 'GeV'};
    Date = [2008 01 01 12 00 00];
    EMFF = {'Bfield', ...
        HelioBfield("NParam",struct(), "UseMeters", 1, "UseTeslas", 1)}; 
    % 'Helio' / {'Helio', [-1 0 0]}
    Steps = {'GTalgo', 'BB', 'TimeStep', 86400, 'Nsteps', 1e5};
    Medium = '';
    InteractEM = '';
    InteractNUC = '';
    GTmag = '';
    SaveMode = {'Coordinates', 'on'};
    BC = {'Rmin', 0.5, 'Rmax', 120, 'Units', 'AU'};
    Verbose = 1;
    IOinfo  = {'SaveDir', '', 'SaveFile', 'test_center'};
    Nevents = 1;

elseif strcmp(Region,'G')
    Ro = {'Center', [0 0 0], 'Radius', 100, 'Units', 'pc'};
    Vo = {'Directions', [0 0 0]};
    Particle = {'Type', 'pr', 'T', 1.0, 'Units', 'EeV'};
    Date = 0;
    EMFF = '';
    % EMFF = {'Bfield', 'J12'};
    Steps = {'GTalgo', 'BB', 'TimeStep', 1*365*86400, 'Nsteps', 1e3};
    Medium = '';
    % Medium = {'ISM', 'MOSDIS'}; 
    InteractEM = '';
    InteractNUC = '';
    SaveMode = {'Coordinates', 'off'}; % 'Zmin', 0.2
    BC = {'Rmin', 1, 'Rmax', 101, 'Units', 'pc'};
    GTmag = '';
    Nevents = 1e6;
    IOinfo  = {'SaveDir', '/home/crteam/Документы/GT', 'SaveFile', 'test_center'};
    Verbose = 1;

elseif strcmp(Region,'P') % Need to be updated
    Ro = {[100 0 0],'km'};
    Vo = [-1 0 0];
    Particle = {'pr', 100, 'MeV'};
    Date = 0;
    EMFF = {'B', {'Dip', 50e16}};
    Steps = {'BB', 1e-6, 'Nsteps', 1e7};
    Medium = '';
    Interaction = '';
    SaveMode = [1 0 0 0 0 0 0];
    BC = {'Rmin', 10, 'Rmax', 1000};
    Verbose = 1;
    
end

GTparam = SetGTparam(Ro, Vo, Particle, Date, EMFF, Region, Steps, Medium, InteractEM, InteractNUC, ...
                                                                SaveMode, BC, GTmag, Nevents, IOinfo, Verbose);
GTclear
GTparticle = SetGTparticle(GTparam);
GTtrack = RunGetTrajectoryInEMField(GTparam, GTparticle);

%% Plot results
%  PlotEarth('Atmo', 'off', 'Scale', GTparam.Ro.Units, 'CRF', GTparam.Ro.CRFotp)
%PlotGalaxy(2)
%Tparam.GTfile.LoadFile = 1;
%PlotGTtrack(GetGTtrack(GTparam, 1, 1))         % NEED TO BE CORRECTED
% PlotGTtrack('GTtrack', GTtrack, 'LineColor', 'Red1')    % FIX TO PLOT ALL EVENTS
PlotGTtrack('GTtrack', GTtrack, 'LineColor', 'Blue1', 'Event', length(GTtrack))