function GTparam = SetGTparam(Ro, Vo, Particle, Date, EMFF, Region, Steps, Medium, InteractEM, InteractNUC, ...
                                                                            SaveMode, BC, GTmag, Nevents, IOinfo, Verbose)
    arguments
        Ro          (1,:)   cell
        Vo          (1,:)   cell
        Particle    (1,:)   cell
        Date        (1,:)   double  {CheckDate(Date)}
        EMFF        (1,:)   cell
        Region      (1,1)   char
        Steps       (1,:)   cell
        Medium      (1,:)   cell
        InteractEM  (1,:)   cell
        InteractNUC (1,:)   cell
        SaveMode    (1,:)   cell
        BC          (1,:)   cell
        GTmag       (1,:)   cell
        Nevents     (1,1)   double  {mustBePositive(Nevents)}
        IOinfo      (1,:)   cell
        Verbose     (1,1)   double  {mustBeMember(Verbose, [0, 1, 2, 3, 10])}
    end

    % NO BFIELD - NO GTMAG
    % Not work unification of variables if SetGTparticle 157-178
    % Special units conversion for BC
    
    if Verbose
        fprintf('   Set GTparam variables                   ...')
    end

    % / Ro /
    RoArgs = GetRoArgs(Ro{:});
    GTparam.Ro = RoArgs;
    GTparam.Ro.input = Ro;

    % / Vo /
    VoArgs = GetVoArgs(Vo{:});
    GTparam.Vo = VoArgs;
    if GTparam.Ro.Radius == 0
        if ~isempty(GTparam.Vo.IO)
            error('Variable IO must be an empty')
        end
    end
    GTparam.Vo.input = Vo;
    
    % / Particle /
    ParticleArgs = GetParticleArgs(Particle{:});
    GTparam.Particle = ParticleArgs;
    GTparam.Particle.input = Particle;
    
    % / Date /
    Dates.Date = Date;
    [Dates.Year, Dates.DoY, Dates.Secs, Dates.DTnum] = deal(0);
    if numel(Dates.Date) ~= 1
        [Dates.Year, Dates.DoY, Dates.Secs, Dates.DTnum] = YDS(Dates.Date);
    end
    GTparam.Dates = Dates;
    
    % / EMFF /
    EMFFArgs = GetEMFFArgs(EMFF{:});
    GTparam.EMFF = EMFFArgs;
    GTparam.EMFF.input = EMFF;
    if GTparam.EMFF.Bpos || GTparam.EMFF.Epos 
        GTparam.EMFF.Units = mUnits;
        GTparam.EMFF.Dates = GTparam.Dates;
    end

    % / Region /
    RegionArgs = GetRegionArgs('Name', Region);
    GTparam.Region = RegionArgs;
    if GTparam.Region.Code ~= 1
        mustBeMember(GTparam.Ro.CRFinp, {''})
        mustBeMember(GTparam.Ro.CRFotp, {''})
    end
    
    % / Steps /
    StepsArgs = GetStepsArgs(Steps{:});
    GTparam.Steps = StepsArgs;
    GTparam.Steps.input = Steps;
    
    % / Medium /
    MediumArgs = GetMediumArgs(Medium{:});
    GTparam.Medium = MediumArgs;
    GTparam.Medium.input = Medium;
    
    % / InteractEM /
    InteractEMArgs = GetInteractEMArgs(InteractEM{:});
    GTparam.InteractEM = InteractEMArgs;
    GTparam.InteractEM.input = InteractEM;    
    if strcmp(GTparam.Steps.GTalgo, 'RK4') || strcmp(GTparam.Steps.GTalgo, 'RK6')
        if EMFFArgs.Epos
            error('RK methods are not supported with acceleration (Efield must be off)')
        end
        if strcmp(GTparam.InteractEM.RadLoss, 'on') || EMFFArgs.Epos
            error('RK methods are not supported with energy losses (RadLoss must be off)')
        end
    end
    
    % / InteractNUC /
    InteractNUCArgs = GetInteractNUCArgs(InteractNUC{:});
    GTparam.InteractNUC = InteractNUCArgs;
    if ~isempty(GTparam.InteractNUC)
        mustBeNonempty(GTparam.Medium)
    end
    GTparam.InteractNUC.input = InteractNUC;
    
    % / SaveMode /
    SaveModeArgs = GetSaveModeArgs(SaveMode{:});
    GTparam.SaveMode = SaveModeArgs;
    GTparam.SaveMode.input = SaveMode;
    
    % / BC /
    BCArgs = GetBCArgs(BC{:});
    GTparam.BC = BCArgs;
    DBC = GTparam.BC.DefaultBCArray;
        DBCcur = [GTparam.Ro.Center + GTparam.Ro.Radius, ...
             norm(GTparam.Ro.Center + GTparam.Ro.Radius)];
        DBCcur = DBCcur*GTparam.Ro.Rsconv;
        BCInputError(1:4) = abs(DBCcur) < DBC(1:4);
        DBCcur = [GTparam.Ro.Center - GTparam.Ro.Radius, ...
             norm(GTparam.Ro.Center - GTparam.Ro.Radius)];      % NEED TO FIX AN ERROR
        DBCcur = DBCcur*GTparam.Ro.Rsconv;
        BCInputError(5:8) = abs(DBCcur) > DBC(6:9);
    %if sum(BCInputError)
    %    error('BCs strikes on input, check BC units')
    %end
    GTparam.BC.input = BC;

    % / GTmag /
    GTmagArgs = GetGTmagArgs(GTmag{:});
    GTparam.GTmag = GTmagArgs;
    if GTparam.GTmag.IsOn
        if strcmp(GTparam.SaveMode.Bfield, 'off')
            error('Variable SaveMode.Bfield must be on')
        end
        if ~strcmp(GTparam.Region.Name, 'M')
            error('GTmag could be calculated for magnetosphere only')
        end
    end
    GTparam.GTmag.input = GTmag;
        
    % / Nevents /
    GTparam.Nevents     = Nevents;
    
    % / IOinfo /
    IOArgs = GetIOArgs(IOinfo{:});
    GTparam.IOinfo = IOArgs;
    GTfile = GetGTfile(GTparam);
    GTparam.GTfile = GTfile;
    GTparam.IOinfo.input = IOinfo;
    
    % / Random seeds for each file /
    for ifile = 1 : IOArgs.Nfiles + 1 * (IOArgs.Nfiles == 0)
        GTparam.RandSeed(ifile) = GetRandomSeed;
        pause(1)
    end

    % / Verbose /
    GTparam.Verbose = Verbose;
    
    if Verbose
        fprintf('   successfully \n')
    end

    % / Save GTparam to file /
    if ~isempty(GTparam.GTfile) && Verbose ~= 10
        if GTparam.Verbose
            fprintf('   Save GTparam to file                    ...')
        end
        if ~isempty(GTparam.GTfile.dir.param) && ~exist(GTparam.GTfile.dir.param, 'dir')
            mkdir(GTparam.GTfile.dir.param)
            mkdir(GTparam.GTfile.dir.particle)
            mkdir(GTparam.GTfile.dir.track)
        end
        save(GTparam.GTfile.file.param, 'GTparam')
        if GTparam.Verbose
            fprintf('   done \n')
        end
    else
        mustBeMember(GTparam.IOinfo.Nfiles, 1)
    end
end

%% / Ro / 
function RoArgs = GetRoArgs(RoArgs)
    arguments
        RoArgs.Center	(1,3)   double                                                                      = [0 0 0]
        RoArgs.LLA	    (1,3)   double                                                                      = [0 0 0]
        RoArgs.Radius	(1,1)   double  {mustBeNonnegative(RoArgs.Radius)}                                  = 0
		RoArgs.Units	        char    {mustBeMember(RoArgs.Units, ...
                                                     {'m', 'km', 'RE', 'AU', 'pc', 'kpc', 'ddkm'})}         = ''
        RoArgs.CRFinp	        char    {mustBeMember(RoArgs.CRFinp, {'','geo', 'mag', 'gsm'})}             = ''
        RoArgs.CRFotp           char    {mustBeMember(RoArgs.CRFotp, {'','geo', 'mag', 'gsm'})}             = ''
        RoArgs.Rsconv           char                                                                        = ''
    end

    if sum(RoArgs.LLA)
        if sum(RoArgs.Center)
            error('Define Center or LLA variable, not both')
        end
        if ~strcmp(RoArgs.Units, 'ddkm')
            error('Wrong LLA units')
        end
        if ~isempty(RoArgs.CRFinp) && ~strcmp(RoArgs.CRFinp, 'geo')
            error('Not possible to use LLA out of GEO RF')
        end
        RoArgs.Center = lla2ecef([RoArgs.LLA(1) RoArgs.LLA(2) RoArgs.LLA(3)*1000]);
        RoArgs.Units  = 'm';
    end

    lUnits2m = getlUnits2m;
    RoArgs.Rsconv = lUnits2m(RoArgs.Units);
end

%% / Vo / 
function VoArgs = GetVoArgs(VoArgs)
    arguments
        VoArgs.Directions	(1,3)   double                                                  		= [0 0 0]
        VoArgs.Pitch        (1,1)   double                                                  		= 1000
        VoArgs.IO                   char    {mustBeMember(VoArgs.IO, {'', 'In', 'Out'})}			= ''
        VoArgs.Isotropic    (1,1)   double                                                  		= 1
        VoArgs.Backtracing          char   {mustBeMember(VoArgs.Backtracing, {'', 'on', 'off'})}   	= 'off'
    end
    if max(abs(VoArgs.Directions))
        VoArgs.Isotropic = 0;
        VoArgs.Directions = VoArgs.Directions/norm(VoArgs.Directions);
        if VoArgs.Pitch ~= 1000
            error('Define Directions or Pitch variable, not both')
        end
    end
    if VoArgs.Pitch ~= 1000
        validateattributes(VoArgs.Pitch, {'double'}, {'>=', 0, '<=', 90})
        VoArgs.Isotropic = 0;
        if ~isempty(VoArgs.IO)
            error('Variable Pitch incompartible with In/Out flag')
        end
    end
    if ~isempty(VoArgs.IO)
        if VoArgs.Isotropic ~= 1
            error('Variable IO operates with an isotropic flux')
        end  
    end
end

%% / Particle / 
function ParticleArgs = GetParticleArgs(ParticleArgs)
    arguments
        ParticleArgs.Type           char                                                                    = ''
        ParticleArgs.T      (1,1)   double  {mustBeNonnegative(ParticleArgs.T)}                             = 0
		ParticleArgs.Tmin   (1,1)   double  {mustBeNonnegative(ParticleArgs.Tmin)}                          = 0
        ParticleArgs.Tmax   (1,1)   double  {mustBeNonnegative(ParticleArgs.Tmax)}                          = inf
        ParticleArgs.Rig    (1,1)   double                                                                  = 0
		ParticleArgs.Rigmin (1,1)   double  {mustBeNonnegative(ParticleArgs.Rigmin)}                        = 0
        ParticleArgs.Rigmax (1,1)   double  {mustBeNonnegative(ParticleArgs.Rigmax)}                        = inf
        ParticleArgs.Units          char    {mustBeMember(ParticleArgs.Units, ...
                                            {'eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV', 'EeV', 'ZeV', ...
                                              'V',  'kV',  'MV',  'GV',  'TV',  'PV',  'EV',  'ZV'})}       = ''
        ParticleArgs.Sbas           char    {mustBeMember(ParticleArgs.Sbas, {'','T', 'R', 'E', 'F'})}      = ''
        ParticleArgs.Sind	(1,1)   double                                                                  = 0
        ParticleArgs.Mono	(1,1)   double                                                                  = 1
        ParticleArgs.kes            char                                                                    = ''
    end

    mustBeGreaterThanOrEqual(ParticleArgs.Tmax, ParticleArgs.Tmin)
    if ParticleArgs.T > 0
        mustBeMember(ParticleArgs.Tmin, 0)
        mustBeMember(ParticleArgs.Tmax, inf)
    end
    mustBeGreaterThanOrEqual(ParticleArgs.Rigmax, ParticleArgs.Rigmin)
    if ParticleArgs.Rig > 0
        mustBeMember(ParticleArgs.Rigmin, 0)
        mustBeMember(ParticleArgs.Rigmax, inf)
    end

    if strcmp(ParticleArgs.Sbas, 'F')
        mustBeMember(ParticleArgs.Type, {''})
        validateattributes(ParticleArgs.Sind, {'double'}, {'>', 0.1, '<', 5})
    end
    if ParticleArgs.Tmin || ParticleArgs.Rigmin
        ParticleArgs.Mono = 0;
    end
    %{
    if ~ParticleArgs.Rig && ~ParticleArgs.T
        error('Define Rigidity or KineticEnergy variable, one of them')
    end
    %}
    if ParticleArgs.Rig && ParticleArgs.T
        error('Define Rigidity or KineticEnergy variable, not both') 
    end
    if ParticleArgs.Rig
        ParticleArgs.RTs = 'R'; 
    else
        ParticleArgs.RTs = 'T';
    end
    ParticleArgs.RT = ParticleArgs.Rig + ParticleArgs.T;
    ParticleArgs.RTmin = ParticleArgs.Rigmin + ParticleArgs.Tmin;
    if ~isinf(ParticleArgs.Rigmax)
        ParticleArgs.RTmax = ParticleArgs.Rigmax;
    elseif ~isinf(ParticleArgs.Tmax)
        ParticleArgs.RTmax = ParticleArgs.Tmax;
    else
        ParticleArgs.RTmax = inf;
    end

    RUnits = {'V', 'kV', 'MV', 'GV', 'TV', 'PV', 'EV', 'ZV'};
    if ParticleArgs.Rig && ~ismember(ParticleArgs.Units, RUnits)
        error('Invalid value for ''Units'' argument. Value must be a member of this set: \n V / kV / MV / GV / TV / PV / EV / ZV')        
    end
    TUnits = {'eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV', 'EeV', 'ZeV'};
    if ParticleArgs.T && ~ismember(ParticleArgs.Units, TUnits)
        error('Invalid value for ''Units'' argument. Value must be a member of this set: \n eV / keV / MeV / GeV / TeV / PeV / EeV / ZeV')
    end

    % Scaling map for energy units
    UeV2MeV = containers.Map( ...
        {'eV', 'keV', 'MeV', 'GeV', 'TeV', 'PeV', 'EeV', 'ZeV',  'V',  'kV',  'MV',  'GV',  'TV',  'PV',  'EV',  'ZV'}, ...
        {1e-6,  1e-3,     1,   1e3,   1e6,   1e9,  1e12,  1e15, 1e-6,  1e-3,     1,   1e3,   1e6,   1e9,  1e12,  1e15} ...
    );
    ParticleArgs.kes = UeV2MeV(ParticleArgs.Units);
end

%% / Date /
function CheckDate(Date)
    % Number of elements
    mustBeMember(numel(Date), [1, 3, 6])
    % Check non negative date
    mustBeNonnegative(Date)
    % Check 0 date
    if numel(Date) == 1
        mustBeMember(Date, 0)
    end
end

%% / EMFF / 
function EMFFArgs = GetEMFFArgs(EMFFArgs)
    arguments
        EMFFArgs.Bpos       (1,1)   = 0
        EMFFArgs.Bfield             = 'Null'
		EMFFArgs.Epos       (1,1)   = 0
        EMFFArgs.Efield             = 'Null'
    end
    if ~strcmp(EMFFArgs.Bfield, 'Null')
        EMFFArgs.Bpos = 1;
    end
    if ~strcmp(EMFFArgs.Efield, 'Null')
        EMFFArgs.Epos = 1;
    end
end

%% / Region / 
function RegionArgs = GetRegionArgs(RegionArgs)
    arguments
        RegionArgs.Name           char   {mustBeMember(RegionArgs.Name, {'M', 'H', 'G', 'P'})}
        RegionArgs.Code   (1,1)   double {mustBeNonnegative(RegionArgs.Code)}                   = 0
    end

    Reg2RegCode = containers.Map( ...
        {'M', 'H', 'G', 'P'}, ...
        {  1,   2,   3,   4} ...
    );
    RegionArgs.Code = Reg2RegCode(RegionArgs.Name);    
end


%% / Steps / 
function StepsArgs = GetStepsArgs(StepsArgs)
    arguments
        StepsArgs.GTalgo                char   {mustBeMember(StepsArgs.GTalgo, {'BB', 'RK4', 'RK6'})}   = 'BB'
        StepsArgs.TimeStep      (1,1)   double {mustBeNonnegative(StepsArgs.TimeStep)}                  = 0
        StepsArgs.NLarmor       (1,1)   double {mustBeNonnegative(StepsArgs.NLarmor)}     				= 0
		StepsArgs.Time          (1,1)   double {mustBeNonnegative(StepsArgs.Time)}                      = inf
        StepsArgs.Nsteps        (1,1)   double {mustBeNonnegative(StepsArgs.Nsteps)}                    = inf
        StepsArgs.Generation    (1,1)   double {mustBeNonnegative(StepsArgs.Generation)}                = 0
        StepsArgs.Tstart        (1,1)   double {mustBeNonnegative(StepsArgs.Tstart)}                    = 0
    end
    if ~isinf(StepsArgs.Nsteps) && ~isinf(StepsArgs.Time)
        error('Define Nsteps or Time variable, not both')
    end
    if isinf(StepsArgs.Nsteps) && isinf(StepsArgs.Time)
        error('Define Nsteps or Time variable, one of them')
    end
    if isinf(StepsArgs.Nsteps)
        StepsArgs.Nsteps = StepsArgs.Time/StepsArgs.TimeStep;
    end
    if StepsArgs.NLarmor && StepsArgs.TimeStep
        error('Define TimeStep or NLarmor variable, not both') 
    end
    if StepsArgs.NLarmor == 0 && StepsArgs.TimeStep == 0
        error('TimeStep not defined')
    end
end

%% / Medium / 
function MediumArgs = GetMediumArgs(MediumArgs)
    arguments
        MediumArgs.Atmo     char   {mustBeMember(MediumArgs.Atmo, {'', 'NRLMSISE-00'})}     = ''
        MediumArgs.Iono     char   {mustBeMember(MediumArgs.Iono, {'', 'IRI-2016'})}        = ''
		MediumArgs.ISM      char   {mustBeMember(MediumArgs.ISM,  {'', 'MOSDIS'})}          = ''
    end
end

%% / InteractEM / 
function InteractEMArgs = GetInteractEMArgs(InteractEMArgs)
    arguments
        InteractEMArgs.RadLoss              char            {mustBeMember(InteractEMArgs.RadLoss, {'', 'on', 'off'})}       = 'off'
        InteractEMArgs.UseFun       (1,1)   double                                                                          = 0
        InteractEMArgs.UserEMIFun           function_handle
    end
    if isfield(InteractEMArgs, 'UserEMIFun')
        InteractEMArgs.UseFun = 1;
    end
end

%% / InteractNUC / 
function InteractNUCArgs = GetInteractNUCArgs(InteractNUCArgs)
    arguments
        InteractNUCArgs.IntPathDen  (1,1)   double {mustBeNonnegative(InteractNUCArgs.IntPathDen)}       	= 0
        InteractNUCArgs.EAS                 char   {mustBeMember(InteractNUCArgs.EAS, {'', 'on', 'off'})}   = 'off'
        InteractNUCArgs.GenMax      (1,1)   double {mustBeNonnegative(InteractNUCArgs.GenMax)}              = Inf
        InteractNUCArgs.EnMin       (1,1)   double {mustBeNonnegative(InteractNUCArgs.EnMin)}               = Inf
        InteractNUCArgs.Exclude                                                                             = ''
        InteractNUCArgs.UseFun      (1,1)   double                                                          = 0
        InteractNUCArgs.UserNUCIFun         function_handle
    end    
    if strcmp(InteractNUCArgs.EAS, 'on') || ~isinf(InteractNUCArgs.GenMax) || ...
        ~isinf(InteractNUCArgs.EnMin) || ~isempty(InteractNUCArgs.Exclude) || isfield(InteractNUCArgs, 'UserNUCIFun')
        error('IntPathDen must be defined')
    end
    if isfield(InteractNUCArgs, 'UserNUCIFun')
        InteractNUCArgs.UseFun = 1;
    end
end

%% / SaveMode / 
function [SaveMode] = GetSaveModeArgs(SaveMode)
    arguments   
        SaveMode.Coordinates    char   {mustBeMember(SaveMode.Coordinates, {'on', 'off'})}  = 'off'
        SaveMode.Clocks         char   {mustBeMember(SaveMode.Clocks, {'on', 'off'})}       = 'off'
		SaveMode.Bfield         char   {mustBeMember(SaveMode.Bfield, {'on', 'off'})}       = 'off'
        SaveMode.Efield         char   {mustBeMember(SaveMode.Efield, {'on', 'off'})}       = 'off'
		SaveMode.Energy         char   {mustBeMember(SaveMode.Energy, {'on', 'off'})}       = 'off'
        SaveMode.Angles         char   {mustBeMember(SaveMode.Angles, {'on', 'off'})}       = 'off'
        SaveMode.TrackInfo      char   {mustBeMember(SaveMode.TrackInfo, {'on', 'off'})}    = 'off'
        SaveMode.ind      (1,7) double = [0 0 0 0 0 0 0]
        SaveMode.UserSaveGTCondition    function_handle
        SaveMode.UseFunction	double 			= 0
    end
    i = 0;
    for fn = fieldnames(SaveMode)'
        i = i + 1;
        if strcmp(SaveMode.(fn{1}), 'on')
            SaveMode.ind(i) = 1;
        end
    end
    SaveMode.UseFunction = isfield(SaveMode, 'UserSaveGTCondition');
end

%% / BC /
function BCArgs = GetBCArgs(BCArgs)
    arguments 
		BCArgs.Xmin				double 			= 0
		BCArgs.Ymin	        	double 			= 0
		BCArgs.Zmin	        	double 			= 0
        BCArgs.Rmin	        	double 			= 0
		BCArgs.Dist2Path		double 			= 0
		BCArgs.Xmax	        	double 			= Inf
		BCArgs.Ymax	        	double 			= Inf
		BCArgs.Zmax	        	double 			= Inf
		BCArgs.Rmax	        	double 			= Inf
		BCArgs.MaxPath	    	double 			= Inf
		BCArgs.MaxPathDen		double 			= Inf
		BCArgs.MaxTime	    	double 			= Inf
        BCArgs.MaxRev	    	double 			= Inf
        BCArgs.UseDefault		double 			= 0
        BCArgs.UseFunction		double 			= 0
		BCArgs.Units	        char    {mustBeMember(BCArgs.Units, {'m', 'km', 'RE', 'AU', 'pc', 'kpc'})} = ''
        BCArgs.UserBCFunction   function_handle
    end
    
    if isfield(BCArgs, 'UserBCFunction')
        DefaultFields = struct2cell(structfun(@(x) 1 - (isinf(x) | x == 0), ...
                                    rmfield(BCArgs, {'UserBCFunction', 'UseDefault', 'UseFunction', 'Units'}), ...
                                    'uni', 0));
    else
        DefaultFields = struct2cell(structfun(@(x) 1 - (isinf(x) | x == 0), ...
                                    rmfield(BCArgs, {'Units'}), ...
                                    'uni', 0));
    end
    BCArgs.DefaultFields = ~[DefaultFields{:}];

    BCArgs.UseDefault       = any(sum([DefaultFields{:}]));
    BCArgs.UseFunction      = isfield(BCArgs, 'UserBCFunction');
    
    if BCArgs.UseFunction
        BCfilename = [func2str(BCArgs.UserBCFunction) '.m'];
        if ~exist(BCfilename, 'file')
            error(['The following files do not exist: ' BCfilename])
        end
        if strcmp(func2str(BCArgs.UserBCFunction), 'BCFullRevolution')
            error('Wrong name of UserBCFunction: BCFullRevolution is an internal GT function (The Number of full revolutions)')
        end
    end
    
    DefaultBCNames = [ ...
    ... %1      2      3      4      5         (Min)
        "Xmin","Ymin","Zmin","Rmin","Dist2Path", ...
    ... %6       7       8       9       10         11            12         13         (Max)
        "Xmax", "Ymax", "Zmax", "Rmax", "MaxPath", "MaxPathDen", "MaxTime", "MaxRev"];
    BCArgs.DefaultBCNames = DefaultBCNames;
    DefaultBCStruct = SubStructByFields(BCArgs, DefaultBCNames);
    BCArgs.DefaultBCArray = struct2array(DefaultBCStruct);
    lUnits2m = getlUnits2m;
    Rsconv = lUnits2m(BCArgs.Units);
    idx = [1:4 6:10];
    BCArgs.DefaultBCArray(idx) = BCArgs.DefaultBCArray(idx) * Rsconv;
    
    function SubStruct = SubStructByFields(S, Fields)
        SubStruct = struct();
        for Field = Fields
            SubStruct.(Field) = S.(Field);
        end
    end
end

%% / GTmag / 
function GTmagArgs = GetGTmagArgs(GTmagArgs)
    arguments
        GTmagArgs.TrackParam                    char   {mustBeMember(GTmagArgs.TrackParam, {'on', 'off'})}      = 'off'
        GTmagArgs.ParticleOrigin                char   {mustBeMember(GTmagArgs.ParticleOrigin, {'on', 'off'})}  = 'off'
        GTmagArgs.IsOn                  (1,1)   double {mustBeMember(GTmagArgs.IsOn, [0 1])}                    = 0
        GTmagArgs.TrackParamIsOn        (1,1)   double {mustBeMember(GTmagArgs.TrackParamIsOn, [0 1])}          = 0
        GTmagArgs.ParticleOriginIsOn    (1,1)   double {mustBeMember(GTmagArgs.ParticleOriginIsOn, [0 1])}      = 0
        GTmagArgs.IsFirstRun            (1,1)   double {mustBeMember(GTmagArgs.IsFirstRun, [0 1])}              = 1
    end
    if strcmp(GTmagArgs.ParticleOrigin, 'on')
        GTmagArgs.ParticleOriginIsOn = 1;
        GTmagArgs.TrackParam = 'on';
        GTmagArgs.TrackParamIsOn = 1;
    end
    if strcmp(GTmagArgs.TrackParam, 'on')
        GTmagArgs.TrackParamIsOn = 1;
    end
    if GTmagArgs.ParticleOriginIsOn || GTmagArgs.TrackParamIsOn
        GTmagArgs.IsOn = 1;
    end
end

%% / IOinfo / 
function IOArgs = GetIOArgs(IOArgs)
    arguments
        IOArgs.Nfiles       (1,1)   double  {mustBeNonnegative(IOArgs.Nfiles)}                  = 1
        IOArgs.SaveDir              char                                                        = ''
		IOArgs.SaveFile             char                                                        = ''
        IOArgs.LoadGTtrack          char    {mustBeMember(IOArgs.LoadGTtrack, {'on', 'off'})}   = 'off'
    end
end

%% / Scaling map for length units /
function lUnits2m = getlUnits2m
    Units = mUnits;
    lUnits2m = containers.Map( ...
        {'m',       'km',       'RE',       'AU',       'pc',       'kpc'}, ...
        {  1, Units.km2m, Units.RE2m, Units.AU2m, Units.pc2m, Units.kpc2m} ...
    );
end