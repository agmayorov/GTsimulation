function GTtrack = GetTrajectoryInEMField(GTparam, GTparticle, GTTrack_in)
%   GTTrack = GetTrajectoryInEMField(GTparam)
%   GTTrack = GetTrajectoryInEMField(GTparam, GTTrack)
% 	Reconstruction of particle's trajectories in electromagnetic fields
%   Ver. 8, red. 0 / April 2023 / A.G. Mayorov, V.S. Golubkov / CRTeam / NRNU MEPhI, Russia
%
%   ==========
%
%   Input/Output:
%       See documentation - https://spacephysics.mephi.ru/GTsimulation/GTsimulation.pdf
%
%   ==========
%
%   Setup Software:
% 		SetGetTrajectoryInEMField.m
%       RunGetTrajectoryInEMField.m
%       SetGTparam.m
%       SetGTparticle.m
%
%   ==========

    %% ==================================================================================================
    %  ----- >>>>> Constants & Units
    %  ==================================================================================================
    
    Units = mUnits;
    Const = mConstants;

    %% ==================================================================================================
    %  ----- >>>>> Read input
    %  ==================================================================================================
    verb = GTparam.Verbose;
    if verb
        if verb > 1
            fprintf('\n')
        end
        fprintf(['   GetTrajectoryInEMField simulation: File No ' ...
                 num2str(GTparam.No.File) ' / Event No ' num2str(GTparam.No.Event) '\n'])
    end

    if strcmp(GTparam.Vo.Backtracing, 'on')
        Backtracing = 1;
        GTparticle_ini = GTparticle;
        GTparticle = GetGTantiparticle(GTparticle);
        GTparticle.Vo = - GTparticle.Vo;
        if verb > 1
            fprintf('   BackTracing mode is ON \n')
            fprintf('   Redefinition of particle on antiparticle \n')
        end
    else
        Backtracing = 0;
        if verb > 1
            fprintf('   BackTracing mode is OFF \n')
        end
    end

    if verb > 1
        fprintf('   Read input ... \n')
    end

    rng(GTparam.RandSeed(GTparam.No.File));
    if verb > 1
        fprintf(['      + Random number generator with seed ' num2str(GTparam.RandSeed) ' \n'])
    end

    % ----------------------------------
    % >>> Initial position / units / CRF
    % ----------------------------------
    if verb > 1
        fprintf('      + Initial positions, kinematics and particles: \n')
    end

    Ro =  GTparticle.Ro;
    Rs =  GTparam.Ro.Units;
    Rsconv = GTparam.Ro.Rsconv;
    Ro = Ro*Rsconv;
    
    % Coordinate reference frame
    CRFin  = GTparam.Ro.CRFinp;
    CRFout = GTparam.Ro.CRFotp;
    if verb > 1
        fprintf(['         Ro: ' num2str(Ro(1)/Rsconv) ' ' num2str(Ro(2)/Rsconv) ...
                             ' ' num2str(Ro(3)/Rsconv) ' ' Rs ' \n'])
        if ~isempty(CRFin)
            fprintf(['         Input/output CRF: ' upper(CRFin) ' / ' upper(CRFout) ' \n'])
        end
    end

    % ----------------------------------
    % >>> Initial direction
    % ----------------------------------
    Vo = GTparticle.Vo;
    if verb > 1
        fprintf(['         Vo: ' num2str(Vo(1))     ' ' num2str(Vo(2))     ' ' num2str(Vo(3))     ' \n'])
        if Backtracing        
        fprintf(['        [Vo: ' num2str(GTparticle_ini.Vo(1))     ' ' num2str(GTparticle_ini.Vo(2))     ' ' ...
            num2str(GTparticle_ini.Vo(3))     ']\n'])
        end
    end

    % ----------------------------------
    % >>> Particle characteristics
    % ----------------------------------
    PDG =  GTparticle.PDG;                
    M   =  GTparticle.M;
    Q   =  GTparticle.Q;
    
    % Nuclear interactions for particles nor antiparticles from Li-6
    if abs(PDG) >= 1000030060
        PDG = abs(PDG);                  
    end

    % Neutral particles
    NeutralParticlesPDG = [22 2112 12 14 16];
    IsNeutralParticle = 0;
    if ismember(abs(PDG), NeutralParticlesPDG)
        IsNeutralParticle = 1;
    end

    % Kinetic energy
    To = GTparticle.T;                   % Kinetic energy
    Ts = GTparam.Particle.Units;         % Kinetic energy scale
    kes = GTparam.Particle.kes;          % Scaling for energy units 

    % Total energy and rigidity (just for verb, not used by programm)
    To = To*kes;
    E = To + M;                               % Total energy of particle in MeV
    V = Const.c*sqrt(E^2-M^2)/(To+M);         % Velosity of particle in m/s
    Rig = sqrt(E^2-M^2)/abs(Q/Const.e)/1e3;   % Rigidity in GV

    if verb > 1
        if ~isempty(GTparticle.Type)
            fprintf(['         Particle: ' GTparticle.Type{:} ' (M = ' num2str(M) ' MeV, Z = ' num2str(round(Q/Const.e))  ') \n'])
            if Backtracing
            fprintf(['        [Particle: ' GTparticle_ini.Type{:} ' (M = ' num2str(GTparticle_ini.M) ' MeV, Z = ' ...
                num2str(round(GTparticle_ini.Q/Const.e))  ')]\n'])
            end
        else
            fprintf(['         Particle: PDG ' num2str(PDG) ', M = ' num2str(M) ' MeV, Z = ' num2str(round(Q/Const.e))  ' \n'])
            if Backtracing
            fprintf(['        [Particle: PDG ' num2str(GTparticle_ini.PDG) ', M = ' num2str(GTparticle_ini.M) ' MeV, Z = ' ...
                num2str(round(Q/Const.e))  ']\n'])
            end
        end
        if IsNeutralParticle
            fprintf('         IsNeutralParticle: yes \n')
        else
            fprintf('         IsNeutralParticle: no \n')
        end
        fprintf(['         Kin. energy: ' num2str(To/kes) ' ' Ts ' / Rig.: ' num2str(Rig) ' GV \n'])
        fprintf(['         Velosity: ' num2str(V) ' m/s \n'])
        fprintf(['         beta: ' num2str(V/Const.c) ' \n'])
    end

    % ----------------------------------
    % >>> Date
    % ----------------------------------
    if verb > 1
        fprintf('      + Date and Region: \n')
    end
    
    Dates = GTparam.Dates;

    if verb > 1
        if Dates.Date == 0
            fprintf('         Date: Empty \n')
        else
            fprintf(['         Date (YYYY MM DD HH:MM:SS): ' ...
                num2str(Dates.Date(1)) ' ' num2str(Dates.Date(2)) ' ' num2str(Dates.Date(3)) ' ' ...
                num2str(Dates.Date(4)) ':' num2str(Dates.Date(5)) ':' num2str(Dates.Date(6)) ' \n'])
            fprintf(['         Date (Year DoY Secs): ' ...
                num2str(Dates.Year) ' ' num2str(Dates.DoY) ' ' num2str(Dates.Secs) ' \n']) 
        end
    end

    % ----------------------------------
    % >>> Region
    % ----------------------------------
    RegionCode = GTparam.Region.Code;

    if verb > 1
        fprintf(['         Region: ' GTparam.Region.Name ' / ' num2str(RegionCode) '\n'])
    end

    % ----------------------------------
    % >>> Coordinate reference frame
    % ----------------------------------
    if ~isempty(CRFin) && ~strcmp(CRFin, 'geo')
        switch lower(CRFin)
            case 'dipmag'
                [Ro(1), Ro(2), Ro(3)] = geo2dipmag(Ro(1), Ro(2), Ro(3), GTparam.EarthBfield.GTB.DIP.psi, -1);
                [Vo(1), Vo(2), Vo(3)] = geo2dipmag(Vo(1), Vo(2), Vo(3), GTparam.EarthBfield.GTB.DIP.psi, -1);
            case 'mag'
                [Ro(1), Ro(2), Ro(3)] = geo2mag_eccentric(Ro(1), Ro(2), Ro(3), -1, Dates.DTnum);
                [Vo(1), Vo(2), Vo(3)] = geo2mag(Vo(1), Vo(2), Vo(3), -1);
            case 'gsm'
                [Ro(1), Ro(2), Ro(3)] = geo2gsm(Ro(1), Ro(2), Ro(3), ...
                                        Dates.Year, Dates.DoY, Dates.Secs, -1);
                [Vo(1), Vo(2), Vo(3)] = geo2gsm(Vo(1), Vo(2), Vo(3), ...
                                        Dates.Year, Dates.DoY, Dates.Secs, -1);
        end
        if verb > 1
            fprintf('      + Coordinates in GEO: \n')
            fprintf(['         Ro: ' num2str(Ro(1)/Rsconv) ' ' num2str(Ro(2)/Rsconv) ...
                                 ' ' num2str(Ro(3)/Rsconv) ' ' Rs ' \n'])
            fprintf(['         Vo: ' num2str(Vo(1))     ' ' num2str(Vo(2))     ' ' num2str(Vo(3))     ' \n'])
        end
    end

    % ----------------------------------
    % >>> EM Field Function
    % ----------------------------------
	UseEMfield = 0;
    if GTparam.EMFF.Bpos || GTparam.EMFF.Epos
		UseEMfield = 1;
		if verb > 1
		    fprintf('      + Electromagnetic field functions: \n')
		end
    end
    
    % M Field Function settings
    if GTparam.EMFF.Bpos
        GTB = GTparam.EMFF.Bfield;
        if verb > 1
            fprintf(['         Magnetic field in:' GTparam.EMFF.Bfield.Region])
        end
    else
        if verb > 1
            fprintf('        Without magnetic field\n')
        end
    end
    GTparam.BC.MaxRevIsOn = 0;
    
%     TODO  
%     if RegionCode == 1 && UseEMfield     
%         if ~isinf(GTparam.BC.MaxRev)
%             GTparam.BC.MaxRevIsOn = 1;
%         end
%         if GTB.GetI || ...                              % Put into igrf13_gh function
%            GTparam.GTmag.ParticleOriginIsOn || ...      % Need number of rotation
%            GTparam.BC.MaxRevIsOn                        % Number of rotations for BC
%             IGRF = GetIGRFcoefs(Dates.DTnum);
%             if GTB.GetI
%                 GTB.IGRF = IGRF;
%             end
%         end
%         if GTB.GetC
%            GTB.CHAOS = GetCHAOScoefs(Dates.Date, ...
%                GTB.CHAOS.IsIntCore, GTB.CHAOS.IsIntCrustal, GTB.CHAOS.IsExt, ...
%                num2str(GTparam.EarthBfield.GTB.CHAOS.ModVer));
%         end
%     end
            
    % E Field Function settings
    %if GTparam.EMFF.Epos
    %end

    % ----------------------------------
    % >>> Time steps
    % ----------------------------------
    if verb > 1
        fprintf('      + Time steps: \n')
    end
    GTalgo = GTparam.Steps.GTalgo;      % Tracing algorithm

    GTalgo2ID = containers.Map( ...
        {'BB', 'RK4', 'RK6'}, ...
        {  1,     2,     3} ...
    );
    GTalgoID = GTalgo2ID(GTalgo);

    dt      = GTparam.Steps.TimeStep;    % Time step in s
    if ~dt
        dt  = GetTimeStep(M, round(Q/Const.e), To, GTparam.Steps.NLarmor);
    end
    Nsteps  = GTparam.Steps.Nsteps;      % Number of steps
    Gen     = GTparam.Steps.Generation;  % 0 for initial particle
    Tstart  = GTparam.Steps.Tstart;      % 0 for initial particle
    if verb > 1
        fprintf(['         Tracing algorithm: ' GTalgo ' (' num2str(GTalgoID) ') \n'])   
        fprintf(['         Time step: ' num2str(dt) ' s \n'])
        fprintf(['         Tot. time: ' num2str(GTparam.Steps.Time) '\n'])
        fprintf(['         Nsteps: ' num2str(Nsteps) ' \n'])
        fprintf(['         beta*dt: ' num2str(V*dt/Rsconv) ' ' Rs ' / ' num2str(V*dt/1e3) ' km \n'])
    end
    GTparam.dt = dt;

    % ----------------------------------
    % >>> Medium
    % ----------------------------------
    if verb > 1
        fprintf('      + Medium characteristics: \n')
    end
    
    UseMedium = 0;
    if ~isempty(GTparam.Medium.input)
        if abs(PDG) == 12 || abs(PDG) == 14 || abs(PDG) == 16
            if verb > 1
                fprintf('         UseMedium: neutrino, skip medium \n')
            end
        else
            UseMedium = 1;
            if verb > 1
                fprintf('         UseMedium: yes \n')
            end
            MediumPar = GTparam.Medium;           
    
            % Atmosphere
            if RegionCode == 1
                if verb > 1
                    fprintf('         Medium: Atmosphere\n')
                end
                [GTM.GetAtmo, GTM.GetIono] = deal(0);
                for i = 1:numel(MediumPar)
                    if strcmp(MediumPar(i), 'Iono')
                        GTM.GetIono = 1;
                        IonoMod = MediumPar{i+1};
                        if verb > 1
                            fprintf(['           - Ionosphere / Model ' IonoMod '\n'])
                        end
                    elseif strcmp(MediumPar(i), 'Atmo')                    
                        GTM.GetAtmo = 1;
                        AtmoMod = MediumPar{i+1};
                        if verb > 1
                            fprintf(['           - Atmosphere / Model ' AtmoMod '\n'])
                        end
                    end
                end
                
            % Heliosphere
            elseif RegionCode == 2
                if verb > 1
                    fprintf('         Medium: Heliosphere\n')
                end
    
            % Galaxy
            elseif RegionCode == 3
                if verb > 1
                    fprintf('         Medium: ISM\n')
                end
                GTM.GetISM = 0;
                if ~isempty(MediumPar.ISM)
                    GTM.GetISM = 1;
                    ISMMod = MediumPar.ISM;
                    if verb > 1
                        fprintf(['           - ISM / Model ' ISMMod '\n'])
                    end
                    if strcmp(ISMMod, 'MOSDIS')
                        GTM.DEN = load('S.mat');  %n(HI) at b = 0, sm^(-3) (Gordon & Burton, 1976)
                    end
                end
    
            % Pulsar
            elseif RegionCode == 4
                if verb > 1
                    fprintf('         Medium: Pulsar\n')
                end
            end
        end        
    else
        if verb > 1
            fprintf('         UseMedium: no \n')
        end
    end

    % ----------------------------------
    % >>> Decay
    % ----------------------------------
    UseDecay = 0;
    if abs(PDG) == 13 || abs(PDG) == 2112 || abs(PDG) == 1000010030
        % mu, neutron, triton
        UseDecay = 1;
        rnd_dec = rand;
        if abs(PDG) == 13
            tau_o = 2.196981e-6;
        elseif abs(PDG) == 2112
            tau_o = 879.4;
        elseif abs(PDG) == 1000010030
            tau_o = (4500*86400)/log(2);
        end
        if verb > 1
            fprintf(['         UseDecay: ' num2str(UseDecay) ' \n'])
            fprintf(['         Decay rnd: ' num2str(rnd_dec) ' \n'])
        end
    else
        if verb > 1
            fprintf('         UseDecay: no \n')
        end
    end

    % ----------------------------------
    % >>> Simulation of interactions
    % ----------------------------------
    if verb > 1
        fprintf('      + Simulation of interactions: \n')
    end
    
    UseInteractEM = 0;      % Electromagnetic interaction
    UseInteractNUC = 0;     % Nuclei inelastic interaction
    
    if ~isempty(GTparam.InteractEM) || ~isempty(GTparam.InteractNUC)
        if abs(PDG) == 12 || abs(PDG) == 14 || abs(PDG) == 16
            if verb > 1
                fprintf('         UseInteraction: (anti)neutrino, skip interaction \n')
            end
        else
            IntPathDen = GTparam.InteractNUC.IntPathDen;
            if IntPathDen
                UseInteractNUC = 1;
            end
            UseEAS = 0;
            if strcmp(GTparam.InteractNUC.EAS, 'on')
                UseEAS = 1;
            end
            EnMin   = GTparam.InteractNUC.EnMin;
            GenMax  = GTparam.InteractNUC.GenMax;
            Exclude = GTparam.InteractNUC.Exclude;
            if ischar(Exclude)
                % Exclude all leptons
                if strcmp(Exclude, 'l')
                    Exclude = 11:18;
                end
            end
            UserNUCIFun = '';
            if GTparam.InteractNUC.UseFun
                UserNUCIFun = GTparam.InteractNUC.UserNUCIFun;
            end

            UseRadLoss = 0;
            if strcmp(GTparam.InteractEM.RadLoss, 'on')
                UseInteractEM = 1;
                UseRadLoss = 1;
                RadLossBT = 1*(Backtracing == 0) + (-1)*(Backtracing == 1);
            end
            UserEMIFun = '';
            if GTparam.InteractEM.UseFun
                UserEMIFun = GTparam.InteractEM.UserEMIFun;
            end

            if verb > 1
                if UseInteractNUC == 1
                    fprintf('         UseInteractNUC: yes \n')
                    fprintf(['           - IntPathDen: ' num2str(IntPathDen) ' g/cm2 \n'])
                    if UseEAS == 1
                        fprintf('           - UseEAS: yes \n')
                    else
                        fprintf('           - UseEAS: no \n')
                    end
                    fprintf(['           - Generation: ' num2str(Gen) ' of ' num2str(GenMax) ' \n'])
                    fprintf(['           - Tstart: ' num2str(Tstart) ', s \n'])
                    fprintf(['           - EnMin: ' num2str(EnMin) ' \n'])
                    if ~isempty(Exclude)
                        fprintf('           - Exclude PDG: \n')
                        fprintf('            ')
                        for i = 1:length(Exclude)
                            fprintf([num2str(Exclude(i)) ' '])
                        end
                        fprintf('\n')
                    else
                        fprintf('           - Exclude PDG: no \n')
                    end
                    if GTparam.InteractNUC.UseFun
                        fprintf(['           - UserNUCIFun: ' UserNUCIFun ' \n'])
                    else
                        fprintf('           - UserNUCIFun: no \n')
                    end
                else
                    fprintf('         UseInteractNUC: no \n')
                end
                if UseInteractEM == 1
                    fprintf('         UseInteractEM: yes \n')
                    if UseRadLoss == 1
                        fprintf('           - RadLoss: yes \n')
                        fprintf(['           - RadLoss sign: ' num2str(RadLossBT) ' \n'])
                    else
                        fprintf('           - RadLoss: no \n')
                    end
                    if GTparam.InteractEM.UseFun
                        fprintf(['           - UserEMIFun: ' UserEMIFun ' \n'])
                    else
                        fprintf('           - UserEMIFun: no \n')
                    end
                else
                    fprintf('         UseInteractEM: no \n')
                end
            end
        end
    else
        if verb > 1
            fprintf('         UseInteraction: no \n')
        end
    end

    % ----------------------------------
    % >>> Save mode
    % ----------------------------------
    if verb > 1
        fprintf('      + Save flags: \n')
    end

    SaveMode = {'Nsave', 'Csave', 'MFFsave', 'EFFsave', 'Tsave', 'Asave', 'Lsave'};
    for i = 1:7
        eval([SaveMode{i} ' = ' num2str(GTparam.SaveMode.ind(i)) ';'])
    end

    if verb > 1
        yesno = containers.Map({0, 1}, {'no', 'yes'});
        if Nsave == 0
            fprintf('         Save flag: first/last point \n')
        else
            fprintf(['         Save flag: each ' num2str(Nsave) ' step \n'])
        end
        fprintf(['         Save clocks: ' yesno(Csave) ' \n'])
        if GTparam.EMFF.Bpos
            fprintf(['         Save M field: ' yesno(MFFsave) ' \n'])
        end
        if GTparam.EMFF.Epos
            fprintf(['         Save E field: ' yesno(EFFsave) ' \n'])
        end
        fprintf(['         Save energy: ' yesno(Tsave) ' \n'])
        fprintf(['         Save angles: ' yesno(Asave) ' \n'])
        fprintf(['         Save trajectory: ' yesno(Lsave) ' \n'])
    end

    % ----------------------------------
    % >>> Break conditions
    % ----------------------------------
    if verb > 1
        fprintf('      + Break conditions: \n')
    end

    BCProps = GTparam.BC;    
    DBC = BCProps.DefaultBCArray;
    ShouldBreakDefault      = false;
    ShouldBreakUserFunction = false;
    StepData = struct();
    DefaultBCExitCause = '';
    UserBCExitCause = '';

    if verb > 1
        if BCProps.UseDefault == 1
            for i = 1:length(BCProps.DefaultFields)
                if ~BCProps.DefaultFields(i)
                    fprintf(['         ' BCProps.DefaultBCNames{i} ': ' ...
                        num2str(GTparam.BC.(BCProps.DefaultBCNames{i})) ' \n'])
                end
            end
        end
        if BCProps.UseFunction == 1 
            fprintf(['         UserFunc: ' func2str(BCProps.UserBCFunction) '.m \n'])
        end
    end

    %  ==================================================================================================
    %% ----- >>>>> Initialization
    %  ==================================================================================================
    if verb > 1
        fprintf('   Initialization ... ')
    end

    % ----------------------------------
    % >>> Output information 
    % ----------------------------------
    if Nsave == 0
        Npts = 2;
    else
        Npts = ceil(Nsteps/Nsave);
    end

    [Xs, Ys, Zs] = deal(zeros(Npts, 1));
    [Hs, Es, Ts, As, Ls] = deal([]);

    if MFFsave == 1
        Hs = zeros(Npts-1, 3);
    end
    if EFFsave == 1
        Es = zeros(Npts-1, 3);
    end
    if Tsave == 1
        Ts = zeros(Npts, 1);
    end
    if Asave == 1
        As = zeros(Npts-1, 1);
    end
    if Lsave == 1
        Ls = zeros(Npts-1, 2);
    end

    [TotTime, TotPathLen, TotPathDen] = deal(0);
    WOut = 0;
    Status = 'ready_to_run';
    
    % ----------------------------------
    % >>> Internal variables
    % ----------------------------------

    IsPrimDecay = 0;
    IsPrimInteract = 0;
 
    [PathLen, PathDen] = deal(0);
    [LocalDen, LocalPathDen, LocalChemComp, nLocal] = deal(0);
    dtsum1 = 0;                         % For DateTimeRedefine
    dtsum2 = 0;                         % For Tsyganenko refresh
    if GTalgoID == 1
        [Hx, Hy, Hz] = deal(0);
    end
    if GTalgoID == 2
        [Hx1, Hy1, Hz1, Hx2, Hy2, Hz2, Hx3, Hy3, Hz3] = deal(0);
    end
    if GTalgoID == 3
        [Hx1, Hy1, Hz1, Hx2, Hy2, Hz2, Hx3, Hy3, Hz3, ...
         Hx4, Hy4, Hz4, Hx5, Hy5, Hz5, Hx6, Hy6, Hz6, ...
         Hx7, Hy7, Hz7] = deal(0);
    end
    [Ex, Ey, Ez] = deal(0);
    Rint = zeros(3,1);
    
    UserParam = [];
    full_revolutions = 0;
    lon_total = 0;

    % ----------------------------------
    % >>> First point declaration
    % ----------------------------------
	X = Ro(1); Vxm = Vo(1)*V;
	Y = Ro(2); Vym = Vo(2)*V;
	Z = Ro(3); Vzm = Vo(3)*V;
    Tpart = To;
    if Asave == 1
	    XYZPrev = [X Y Z];
    end

    % ----------------------------------
    % >>> Parameters for numeric schemes
    % ----------------------------------
    Yp= Tpart / M + 1;
    %BB
    if GTalgoID == 1
        q = dt*Q/2/(M*Units.MeV2kg);
    end
    % RK4 & RK6
    if GTalgoID > 1
        p = Q/(M*Units.MeV2kg*Yp);
    end

    % RK6
    if GTalgoID == 3
        c = [0 1/3 2/3 1/3 1/2 1/2 1];
        b = [11/120 0 27/40 27/40 -4/15 -4/15 11/120];
        a =[0       0       0       0       0       0   0;
            1/3     0       0       0       0       0   0;
            0       2/3     0       0       0       0   0;
            1/12    1/3     -1/12   0       0       0   0;
            -1/16   9/8     -3/16   -3/8    0       0   0;
            0       9/8     -3/8    -3/4    1/2     0   0;
            9/44    -9/11   63/44   18/11   0   -16/11  0];
    end

    % ----------------------------------
    % >>> Create Trace TNode
    % ----------------------------------
    TraceData = struct('Gen', [], 'Track', [], 'Fields', [], ...
                       'GTparam', [], 'GTparticle', [], ...
                       'TrackInfo', [], 'GTmag', [], 'BC', []);
    TraceData.BC.Status = Status;
    GTtrack = TNode(TraceData);
	
    %% ==================================================================================================
    %  ----- >>>>> Main procedure
    %  ==================================================================================================

    % ----------------------------------
    % >>> Save the first point
    % ----------------------------------
    k = 1;
    Xs(k) = X;
    Ys(k) = Y; 
    Zs(k) = Z;
    Ts(k) = To;

    % ----------------------------------
    % >>> Start tracing procedure
    % ----------------------------------
    if verb > 1
        fprintf('ok \n   Start tracing procedure \n')
        fprintf('   Progress:  ')
    end
    
    %tic
    for istep = 1 : Nsteps
        if verb > 1
            if rem(istep, fix(Nsteps/min(10,Nsteps))) == 0
                fprintf('%u%%  ', fix(100*istep/Nsteps));
            end
        end

        % ----------------------------------------------
        % ----------------------     Get new coordinates
        % ----------------------------------------------
        
        if ~isempty(GTTrack_in)
        	X = GTTrack_in.data.X(i+1);
		    Y = GTTrack_in.data.Y(i+1);
		    Z = GTTrack_in.data.Z(i+1);

        else
            if ~UseEMfield || IsNeutralParticle
                PathLen = dt*V;
    		    XYZPrev = [X Y Z];
	            X = Vxm*dt + X;
	            Y = Vym*dt + Y;
	            Z = Vzm*dt + Z;
	            XYZCur = [X Y Z];
            else
                PathLen = dt*sqrt(Vxm^2+Vym^2+Vzm^2);

                % .............................................................................
		        % ----------------------------------------------     Get electro-magnetic field
                % .............................................................................
                if GTparam.EMFF.Bpos > 0
                    if GTalgoID == 1
		                [Hx, Hy, Hz] = GTB.GetBfield(X, Y, Z, TotTime);
                        normH = Hx^2+Hy^2+Hz^2;
                    end
                    if GTalgoID == 2
                        [Hx1, Hy1, Hz1] = GTB.GetBfield(X, Y, Z, TotTime);
                        [Hx2, Hy2, Hz2] = GTB.GetBfield(X + Vxm*dt/2, Y + Vym*dt/2, Z + Vzm*dt/2, TotTime);
                        [Hx3, Hy3, Hz3] = GTB.GetBfield(X + Vxm*dt, Y + Vym*dt, Z + Vzm*dt, TotTime);
                        normH = Hx3^2+Hy3^2+Hz3^2;
                    end
                    if GTalgoID == 3
                        [Hx1, Hy1, Hz1] = GTB.GetBfield(X, Y, Z, TotTime);
                        [Hx2, Hy2, Hz2] = GTB.GetBfield(X + Vxm*dt*c(2), Y + Vym*dt*c(2), Z + Vzm*dt*c(2), TotTime);
                        [Hx3, Hy3, Hz3] = GTB.GetBfield(X + Vxm*dt*c(3), Y + Vym*dt*c(3), Z + Vzm*dt*c(3), TotTime);
                        %[Hx4, Hy4, Hz4] = GTB.GetBfield(X + Vxm*dt/2, Y + Vym*dt/2, Z + Vzm*dt/2, TotTime);
                        Hx4 = Hx2; Hy4 = Hy2; Hz4 = Hz2;
                        [Hx5, Hy5, Hz5] = GTB.GetBfield(X + Vxm*dt*c(5), Y + Vym*dt*c(5), Z + Vzm*dt*c(5), TotTime);
                        %[Hx6, Hy6, Hz6] = GTB.GetBfield(X + Vxm*dt, Y + Vym*dt, Z + Vzm*dt, TotTime);
                        Hx6 = Hx5; Hy6 = Hy5; Hz6 = Hz5;
                        [Hx7, Hy7, Hz7] = GTB.GetBfield(X + Vxm*dt*c(7), Y + Vym*dt*c(7), Z + Vzm*dt*c(7), TotTime);
                        normH = Hx7^2+Hy7^2+Hz7^2;
                    end

                    if normH == 0
                        WOut = 4;
		                GTtrack.data.WOut = WOut;
		                GTtrack.data.Status = 'OutOfMagFieldRegion';
		                break
                    end
                end

		        % Get electric field
	%             if GTparam.EMFF.Epos > 0        
	%             end

                % .............................................................................
		        % ---------------------------------------     Relativistic Buneman-Boris scheme
                % .............................................................................
                if GTalgoID == 1
		            G = Tpart/M + 1;                              % gamma_i
		            
		            Uxi = G * Vxm;                                % u_i   
		            Uyi = G * Vym;   
		            Uzi = G * Vzm;   
	     
		            H = norm([Hx Hy Hz]);
    
		            TT = G * tan(q*H/G);                          % module tau
		            
		            Tx = TT * (Hx/H);                             % tau
		            Ty = TT * (Hy/H);
		            Tz = TT * (Hz/H);
    
		            Ux = (Vym*Tz - Vzm*Ty + 2*q*Ex) + Uxi;        % u'
		            Uz = (Vxm*Ty - Vym*Tx + 2*q*Ez) + Uzi;
		            Uy = (Vzm*Tx - Vxm*Tz + 2*q*Ey) + Uyi;      
    
		            UU = (Tx*Ux + Ty*Uy + Tz*Uz)^2 / Const.c^2;         % u*^2 scalar product
		            YY = (1 + (Ux^2 + Uy^2 + Uz^2) / Const.c^2) ^ 0.5;  % gamma'
		            S = YY^2 - TT^2;                              % sigma
		            
		            Ym = Yp;
		            Yp = (0.5 * (S + (S^2 + 4 * (TT^2 + UU)) ^ 0.5)) ^ 0.5;     % gamma_(i+1)
		            Ya = (Yp + Ym) / 2;
    
		            tt = tan(q*H/Yp);                             % module t
		            
		            tx = tt * (Hx/H);                             % t
		            ty = tt * (Hy/H); 
		            tz = tt * (Hz/H);
		            
		            s = (1 + tt^2)^(-1);                          % s

		            Vxp = s/Yp * (Ux+(Ux*tx+Uy*ty+Uz*tz)*tx+(Uy*tz-Uz*ty));        % Vxplus;
		            Vyp = s/Yp * (Uy+(Ux*tx+Uy*ty+Uz*tz)*ty+(Uz*tx-Ux*tz));
		            Vzp = s/Yp * (Uz+(Ux*tx+Uy*ty+Uz*tz)*tz+(Ux*ty-Uy*tx));
                end
                % .............................................................................

                % .............................................................................
		        % ------------------------------------------     Runge–Kutta method - 4th order
                % .............................................................................
                if GTalgoID == 2
                    Hx = Hx1; Hy = Hy1; Hz = Hz1;

                    kx1 = p * ( Vym*Hz1 - Vzm*Hy1 );
                    ky1 = p * ( Vzm*Hx1 - Vxm*Hz1 );
                    kz1 = p * ( Vxm*Hy1 - Vym*Hx1 );
                    
                    kx2 = p * ( Hz2 * (Vym + dt/2 * ky1) - Hy2 * (Vzm + dt/2 * kz1) );
                    ky2 = p * ( Hx2 * (Vzm + dt/2 * kz1) - Hz2 * (Vxm + dt/2 * kx1) );
                    kz2 = p * ( Hy2 * (Vxm + dt/2 * kx1) - Hx2 * (Vym + dt/2 * ky1) );
                    
                    kx3 = p * ( Hz2 * (Vym + dt/2 * ky2) - Hy2 * (Vzm + dt/2 * kz2) );
                    ky3 = p * ( Hx2 * (Vzm + dt/2 * kz2) - Hz2 * (Vxm + dt/2 * kx2) );
                    kz3 = p * ( Hy2 * (Vxm + dt/2 * kx2) - Hx2 * (Vym + dt/2 * ky2) );
                    
                    kx4 = p * ( Hz3 * (Vym + dt * ky3) - Hy3 * (Vzm + dt * kz3) );
                    ky4 = p * ( Hx3 * (Vzm + dt * kz3) - Hz3 * (Vxm + dt * kx3) );
                    kz4 = p * ( Hy3 * (Vxm + dt * kx3) - Hx3 * (Vym + dt * ky3) );
                    
                    Vxp = dt/6 * (kx1 + 2*kx2 + 2*kx3 + kx4) + Vxm;
                    Vyp = dt/6 * (ky1 + 2*ky2 + 2*ky3 + ky4) + Vym;
                    Vzp = dt/6 * (kz1 + 2*kz2 + 2*kz3 + kz4) + Vzm;

                    Ya = Yp;
                end
                % .............................................................................

                % .............................................................................
		        % ------------------------------------------     Runge–Kutta method - 6th order
                % .............................................................................
                if GTalgoID == 3
                    Hx = Hx1; Hy = Hy1; Hz = Hz1;
                    
                    kx(1) = p * ( Vym*Hz1 - Vzm*Hy1 );
                    ky(1) = p * ( Vzm*Hx1 - Vxm*Hz1 );
                    kz(1) = p * ( Vxm*Hy1 - Vym*Hx1 );
                    
                    kx(2) = p * ( Hz2 * (Vym + a(2,1) * dt * ky(1)) - Hy2 * (Vzm + a(2,1) * dt * kz(1)) );
                    ky(2) = p * ( Hx2 * (Vzm + a(2,1) * dt * kz(1)) - Hz2 * (Vxm + a(2,1) * dt * kx(1)) );
                    kz(2) = p * ( Hy2 * (Vxm + a(2,1) * dt * kx(1)) - Hx2 * (Vym + a(2,1) * dt * ky(1)) );
                    
                    kx(3) = p * ( Hz3 * (Vym + a(3,2) * dt * ky(2)) - Hy3 * (Vzm + a(3,2) * dt * kz(2)) );
                    ky(3) = p * ( Hx3 * (Vzm + a(3,2) * dt * kz(2)) - Hz3 * (Vxm + a(3,2) * dt * kx(2)) );
                    kz(3) = p * ( Hy3 * (Vxm + a(3,2) * dt * kx(2)) - Hx3 * (Vym + a(3,2) * dt * ky(2)) );
                    
                    kx(4) = p * ( Hz4 * (Vym + dt * (a(4,1) * ky(1) + a(4,2) * ky(2) + a(4,3) * ky(3))) ...
                        - Hy4 * (Vzm + dt * (a(4,1) * kz(1) + a(4,2) * kz(2) + a(4,3) * kz(3))) );
                    ky(4) = p * ( Hx4 * (Vzm + dt * (a(4,1) * kz(1) + a(4,2) * kz(2) + a(4,3) * kz(3))) ...
                        - Hz4 * (Vxm + dt * (a(4,1) * kx(1) + a(4,2) * kx(2) + a(4,3) * kx(3))) );
                    kz(4) = p * ( Hy4 * (Vxm + dt * (a(4,1) * kx(1) + a(4,2) * kx(2) + a(4,3) * kx(3))) ...
                        - Hx4 * (Vym + dt * (a(4,1) * ky(1) + a(4,2) * ky(2) + a(4,3) * ky(3))) );
                    
                    kx(5) = p * ( Hz5 * (Vym + dt * (a(5,1) * ky(1) + a(5,2) * ky(2) + a(5,3) * ky(3) + a(5,4) * ky(4))) ...
                        - Hy5 * (Vzm + dt * (a(5,1) * kz(1) + a(5,2) * kz(2) + a(5,3) * kz(3) + a(5,4) * kz(4))) );
                    ky(5) = p * ( Hx5 * (Vzm + dt * (a(5,1) * kz(1) + a(5,2) * kz(2) + a(5,3) * kz(3) + a(5,4) * kz(4))) ...
                        - Hz5 * (Vxm + dt * (a(5,1) * kx(1) + a(5,2) * kx(2) + a(5,3) * kx(3) + a(5,4) * kx(4))) );
                    kz(5) = p * ( Hy5 * (Vxm + dt * (a(5,1) * kx(1) + a(5,2) * kx(2) + a(5,3) * kx(3) + a(5,4) * kx(4))) ...
                        - Hx5 * (Vym + dt * (a(5,1) * ky(1) + a(5,2) * ky(2) + a(5,3) * ky(3) + a(5,4) * ky(4))) );
                    
                    kx(6) = p * ( Hz6 * (Vym + dt * (a(6,1) * ky(1) + a(6,2) * ky(2) + a(6,3) * ky(3) + a(6,4) * ky(4) + a(6,5) * ky(5))) ...
                        - Hy6 * (Vzm + dt * (a(6,1) * kz(1) + a(6,2) * kz(2) + a(6,3) * kz(3) + a(6,4) * kz(4) + a(6,5) * kz(5))) );
                    ky(6) = p * ( Hx6 * (Vzm + dt * (a(6,1) * kz(1) + a(6,2) * kz(2) + a(6,3) * kz(3) + a(6,4) * kz(4) + a(6,5) * kz(5))) ...
                        - Hz6 * (Vxm + dt * (a(6,1) * kx(1) + a(6,2) * kx(2) + a(6,3) * kx(3) + a(6,4) * kx(4) + a(6,5) * kx(5))) );
                    kz(6) = p * ( Hy6 * (Vxm + dt * (a(6,1) * kx(1) + a(6,2) * kx(2) + a(6,3) * kx(3) + a(6,4) * kx(4) + a(6,5) * kx(5))) ...
                        - Hx6 * (Vym + dt * (a(6,1) * ky(1) + a(6,2) * ky(2) + a(6,3) * ky(3) + a(6,4) * ky(4) + a(6,5) * ky(5))) );
                    
                    kx(7) = p * ( Hz7 * (Vym + dt * (a(7,1) * ky(1) + a(7,2) * ky(2) + a(7,3) * ky(3) + a(7,4) * ky(4) + a(7,5) * ky(5) + a(7,6) * ky(6))) ...
                        - Hy7 * (Vzm + dt * (a(7,1) * kz(1) + a(7,2) * kz(2) + a(7,3) * kz(3) + a(7,4) * kz(4) + a(7,5) * kz(5) + a(7,6) * kz(6))) );
                    ky(7) = p * ( Hx7 * (Vzm + dt * (a(7,1) * kz(1) + a(7,2) * kz(2) + a(7,3) * kz(3) + a(7,4) * kz(4) + a(7,5) * kz(5) + a(7,6) * kz(6))) ...
                        - Hz7 * (Vxm + dt * (a(7,1) * kx(1) + a(7,2) * kx(2) + a(7,3) * kx(3) + a(7,4) * kx(4) + a(7,5) * kx(5) + a(7,6) * kx(6))) );
                    kz(7) = p * ( Hy7 * (Vxm + dt * (a(7,1) * kx(1) + a(7,2) * kx(2) + a(7,3) * kx(3) + a(7,4) * kx(4) + a(7,5) * kx(5) + a(7,6) * kx(6))) ...
                        - Hx7 * (Vym + dt * (a(7,1) * ky(1) + a(7,2) * ky(2) + a(7,3) * ky(3) + a(7,4) * ky(4) + a(7,5) * ky(5) + a(7,6) * ky(6))) );
                    
                    Vxp = dt * sum(b.*kx) + Vxm;
                    Vyp = dt * sum(b.*ky) + Vym;
                    Vzp = dt * sum(b.*kz) + Vzm;

                    Ya = Yp;
                end
                % .............................................................................

                % .............................................................................
		        % --------------------------------------------------------     Radiation losses
                % .............................................................................
		        % Radiation losses
		        if UseInteractEM == 0
		            Vxm = Vxp;
		            Vym = Vyp;
		            Vzm = Vzp;
		            
		            Tpart = M * (Yp - 1);
		        else
		            acc = [(Vxp - Vxm)/dt, (Vyp - Vym)/dt, (Vzp - Vzm)/dt];     % Acceleration a(i+1/2)
		            
		            Vn = norm([Vxp + Vxm, Vyp + Vym, Vzp + Vzm]);
		            Vinter = [Vxp + Vxm, Vyp + Vym, Vzp + Vzm]./Vn;             % V(i+1/2) normilized
		            
		            acc_par = dot(acc, Vinter);        % check tolerance, a || should be 0
		            acc_per = sqrt(norm(acc)^2 - acc_par^2);

		            dE = dt * (2/(3*4*pi*8.854187e-12)*Q^2*Ya^4/Const.c^3) * (acc_per^2 + acc_par^2 * Ya^2) / Const.e / 1e6; % MeV
		            
		            Tpart = M * (Yp - 1) - RadLossBT * abs(dE);

		            V = Const.c*sqrt((Tpart+M)^2-M^2)/(Tpart+M);  
		            Vn = norm([Vxp Vyp Vzp]);
		            
		            Vxm = V*Vxp/Vn;
		            Vym = V*Vyp/Vn;
		            Vzm = V*Vzp/Vn;
		        end
                % .............................................................................
            end % ~UseEMfield || IsNeutralParticle
		    %Tpart; if i == 5, break; end

		    TotTime = TotTime + dt;
		    TotPathLen = TotPathLen + PathLen;                % m

		    % New coordinates
		    XYZPrev = [X Y Z];
		    X = Vxm*dt + X;
		    Y = Vym*dt + Y;
		    Z = Vzm*dt + Z;
		    XYZCur = [X Y Z];
        end % Get new coordinates

        % ----------------------------------------------
        % --------     Update time-dep variables (M & H)
        % ----------------------------------------------
        % TODO
%         if numel(Dates.Date) ~= 1 && RegionCode <= 2
%             % Refresh Date every minute
%             dtsum1 = dtsum1 + dt;
%             if dtsum1 > 60
%                 Dates.Date = DateTimeRedefine(Dates.Date, dt);
%                 [Dates.Year, Dates.DoY, Dates.Secs, Dates.DTnum] = YDS(Dates.Date);
%                 dtsum1 = 0;
%             end
%             % Refresh data for Dip & Tsyganenko every hour
%             if UseEMfield && RegionCode == 1
%                 if GTB.GetD + GTB.GetT > 0
%                     dtsum2 = dtsum2 + dt;
%                     if dtsum2 > 60*60
%                         if GTB.GetD == 1
%                             GTB.DIP.magmom = GetEarthDipMagMom(Dates.Date(1:3), 'SI_nT');
%                         end
%                         if GTB.GetI == 1
%                             IGRF = GetIGRFcoefs(DTnum);
%                             GTB.IGRF = IGRF;
%                         end
%                         if GTB.GetT == 1
%                             GTB.TSYGpsit = GetPsi([Dates.Year, Dates.DoY, Dates.Secs]);
%                             GTB.TSYGind = GetTsyganenkoInd(0, GTB.TSYG.ModCode, Dates.Date);
%                         end
%                         dtsum2 = 0;
%                     end
%                 end
%             end
%         end % Update time-dep variables

        % ----------------------------------------------
        % -----------------------------------     Medium
        % ----------------------------------------------
        if UseMedium == 1
            Den = 0;
            ChemComp = zeros(1,14);

            % Atmosphere
            if RegionCode == 1
                if norm([X Y Z]) < 8.378e6
                    LLA = ecef2lla([X Y Z]);
                    LLA(3) = LLA(3)/1e3;
                    % Atmo
                    Na = zeros(1, 7);
                    Den_A = 0;
                    if GTM.GetAtmo == 1
                        [~, ~, Den_A, Na] = Atmosphere(LLA, AtmoMod, Dates.Date);
                    end            
                    % Iono
                    Ni = zeros(1, 6);
                    Den_I = 0;
                    if GTM.GetIono == 1
                        [~, ~, Ni, ~] = Ionosphere(LLA, Dates.Date);
                        Den_I = sum(Ni.*[2.656 2.325 0.167 0.664 5.333 4.983]*1e-23);     % 1/cm3 -> g/cm3
                    end            
                    % Total
                    Den = Den_A + Den_I;
                    ChemComp = [[Na Ni]./sum([Na Ni]) 0];
                end

            % Heliosphere
            elseif RegionCode == 2
                ...

            % Galaxy
            elseif RegionCode == 3
                if GTM.GetISM == 1
                    [Den, ChemComp] = InterStMedium_fast(X/Units.kpc2m, Y/Units.kpc2m, Z/Units.kpc2m, GTM.DEN);
                end

            % Pulsar
            elseif RegionCode == 4
                ...

            end
            if isnan(Den)
                Den = 0;
                ChemComp = zeros(1,14);
            end

            PathDen = PathLen*1e2*Den;                    % Path in g/cm2
            TotPathDen = TotPathDen + PathDen;

            if UseInteractNUC && Den > 0
                LocalDen = LocalDen + Den;
                LocalChemComp = LocalChemComp + ChemCompRedefine(ChemComp);
                LocalPathDen = LocalPathDen + PathDen;
                nLocal = nLocal + 1;
                
                LocalDenAlong(nLocal) = PathDen; %#ok<AGROW> 
                if nLocal == 1
                    LocalCoord(1,:) = XYZPrev;
                end
                LocalCoord(nLocal+1,:) = XYZCur; %#ok<AGROW> 
            end
        end

        % ----------------------------------------------
        % ------------------------------------     Decay
        % ----------------------------------------------
        if UseDecay == 1
            lifetime = tau_o * (Tpart/M + 1);
            if rnd_dec > exp(-TotTime/lifetime)

                % Simulation of products
                eval(['product = G4Decay(' num2str(PDG) ',' num2str(Tpart/1e3) ');'])

                if verb > 2
                    Tsec = 0;
                    for prod = 1:length(product)
                        Tsec = Tsec + product(prod).E;
                    end
                    fprintf('%s%16s%s%2u%s%5.3f%s', 'D  ~ ', 'GTDecay', ' ~ ', length(product), ' (', Tsec, ' GeV)');
                end

                % Construct Rotation Matrix
                rotationMatrix = vecRotMat([0 0 1], [Vxm Vym Vzm]./norm([Vxm Vym Vzm]));
                
                % Vlosity directions and initial coordinates of secondaries in XYZ
                Rint = [X Y Z];
                for prod = length(product):-1:1
                    product(prod).PDG(3) = [];
                    vLocalSec = [product(prod).v(1) product(prod).v(2) product(prod).v(3)];
                    product(prod).v(1) = rotationMatrix(1,:)*vLocalSec';
                    product(prod).v(2) = rotationMatrix(2,:)*vLocalSec';
                    product(prod).v(3) = rotationMatrix(3,:)*vLocalSec';
                    clear vLocalSec
                end

                % Run secondaries from decay
                if Gen < GenMax
                    for prod = 1:length(product)
                        if verb > 2
                            fprintf('\n%s%s%2u%s%2u%s%2u%s%s%10s%s%5.3f%s', ...
                                repmat('   ', 1, Gen+2), 'Gen: ', Gen+1, '  (', prod, ' of ', length(product), ')  ', ' = ', product(prod).ParticleName, ...
                                ' (', product(prod).E, ' GeV) = ');
                        end
                        RunGetTrajectory(GTtrack, TraceData, product, prod, EnMin, Exclude, Rint./Rsconv, Rs, CRF, Date, varargin, Tstart, TotTime, Gen, verb);
                    end
                end

                IsPrimDecay = 1;
            end
        end

        % ----------------------------------------------
        % -----     EM Interactions (rad. losses on top)
        % ----------------------------------------------
%         if UseInteractEM == 1
%         if GTparam.InteractEM.UseFun
%         end
%         end

        % ----------------------------------------------
        % -------------     NUC interactions (not decay)
        % ----------------------------------------------
        if UseInteractNUC == 1 && IsPrimDecay == 0
            if LocalPathDen > IntPathDen
                if verb > 1
                    fprintf('%s  ', 'I');
                end

                % Construct Rotation Matrix & Save velosity before possible interaction
                rotationMatrix = vecRotMat([0 0 1], [Vxm Vym Vzm]./norm([Vxm Vym Vzm]));

                % Run Geant-4 simulation
                eval(['[rLocal, vLocal, Tint, status, process, product] = G4Interaction(' num2str(PDG) ',' num2str(Tpart/1e3) ',' ...
                    num2str(LocalPathDen) ','  num2str(LocalDen/nLocal) ',[' num2str(LocalChemComp./nLocal) ']);'])
                Tsec = 0;
                for prod = 1:length(product)
                    Tsec = Tsec + product(prod).E;
                end

                if ~status
                    % Redefine velosity vector
                    Vxm = rotationMatrix(1,:)*vLocal';
                    Vym = rotationMatrix(2,:)*vLocal';
                    Vzm = rotationMatrix(3,:)*vLocal';

                    % Redefine Velocity due to losses of energy
                    Tint = Tint*1e3;
                    if 0.99*Tint > Tpart 
                        error('  TIEMF: Tafter > Tbefore - What happens during this interaction?!')
                    else
                        Tpart = Tint;
                        Vcorr = c*sqrt(1-(M/(Tint+M))^2)/norm([Vxm Vym Vzm]);
                        Vxm = Vxm*Vcorr;
                        Vym = Vym*Vcorr;
                        Vzm = Vzm*Vcorr;
                    end 
                    clear Tint
                end

                % Exit due to interaction
                if status
                    if verb > 2
                        Tsec = 0;
                        for prod = 1:length(product)
                            Tsec = Tsec + product(prod).E;
                        end
                        fprintf('%s%16s%s%2u%s%5.3f%s', 'S  ~ ', process, ' ~ ', length(product), ' (', Tsec, ' GeV)');
                    end
                    IsPrimInteract = 1;

                    % Cordinate of point of interaction in XYZ
                    cpath = norm(rLocal)*1e2*(LocalDen/nLocal);             % Path in cylinder in g/cm2

                    % Vlosity directions of secondaries in XYZ
                    for prod = length(product):-1:1
                        vLocalSec = [product(prod).v(1) product(prod).v(2) product(prod).v(3)];
                        product(prod).v(1) = rotationMatrix(1,:)*vLocalSec';
                        product(prod).v(2) = rotationMatrix(2,:)*vLocalSec';
                        product(prod).v(3) = rotationMatrix(3,:)*vLocalSec';
                        clear vLocalSec
                    end

                    Rint = LocalCoord(find(cumsum(LocalDenAlong) - cpath > 0, 1, 'first'),:);
                    if isempty(Rint)
                        Rint = LocalCoord(end, :);
                    end
                    if RegionCode == 1
                        % Calculate Lat/Lon/Alt
                        lla = ecef2lla([Rint(1) Rint(2) Rint(3)]);
                        lla(3) = lla(3)/1e3;                                % m -> km
                    else
                        lla = [0 0 0];
                    end
                    Rint = Rint/Rsconv;                                     % m -> input scale
                    PoI.XYZ = Rint;
                    PoI.LLA = lla;
                    if verb > 1
                        if RegionCode == 1
                            fprintf('%s', [' on ' num2str(lla(3)) ' km'])
                        end
                    end

                    % Secondaries loop
                    if Gen < GenMax
                        for prod = 1:length(product)
                            if verb > 2
                                fprintf('\n%s%s%2u%s%2u%s%2u%s%s%10s%s%5.3f%s', ...
                                    repmat('   ', 1, Gen+2), 'Gen: ', Gen+1, '  (', prod, ' of ', length(product), ')  ', ' = ', product(prod).ParticleName, ...
                                    ' (', product(prod).E, ' GeV) = ');
                            end

                            % Calculate angle to zenith
                            angle2zen = 0;
                            if RegionCode == 1
                                u = -[Rint(1) Rint(2) Rint(3)]/norm(Rint);
                                Vxyz = [product(prod).v(1) product(prod).v(2) product(prod).v(3)];
                                angle2zen = atan2(norm(cross(u,Vxyz)),dot(u,Vxyz))*180/pi; % 0 without deflection
                            end

                            if RegionCode > 1 || (RegionCode == 1 && (UseEAS == 0 || lla(3) > 80 || angle2zen > 70))
                                % Run secondaries from interaction
                                RunGetTrajectory(Trace, TraceData, product, prod, EnMin, Exclude, Rint, Rs, CRF, Date, varargin, Tstart, TotTime, Gen, verb);
                            else
                                ProductTrace = FillTrace(Trace, TraceData, product(prod).PDG(1), Gen+1, 'EAS', ExitCause.ExtensiveAirShower);
                                % Run Geant-4 EAS simulation
                                if ismember(EASmodel, {'G4', 'Geant4'}) && product(prod).E < 10e3
                                    if product(prod).E > EnMin && ~ismember(product(prod).PDG(1), Exclude)
                                        if verb > 2
                                            fprintf('%s', ' -> EAS       ');
                                        end

                                        % Construct Rotation Matrix
                                        phi2 = 2*pi*rand;
                                        x2 = sin(angle2zen) * cos(phi2);
                                        y2 = sin(angle2zen) * sin(phi2);
                                        z2 = cos(angle2zen);
                                        rotationMatrix2 = vecRotMat([x2 y2 -z2]./norm([x2 y2 -z2]), [Vxm Vym Vzm]./norm([Vxm Vym Vzm]));
                                        SolMag = GetSolMag(Date);

                                        % Center of coordinate RF for EAS simulation
                                        xyz2cen = lla2ecef([lla(1) lla(2) 0]);

                                        % Run simulation of EAS
                                        [~, ~, ~, albedo] = G4Shower( product(prod).PDG(1), product(prod).E, ...
                                            lla(3), angle2zen, SolMag(1), SolMag(2), lla(1), lla(2), SolMag(4), SolMag(3), SolMag(5) );

                                        if verb > 2
                                            fprintf('%s', [' ' num2str(length(albedo)) ' albedo ']);
                                        end
    
                                        % Vlosity directions of secondaries in XYZ
                                        for prod2 = length(albedo):-1:1
                                            vLocalSec = [albedo(prod2).v(1) albedo(prod2).v(2) albedo(prod2).v(3)];
                                            albedo(prod2).v(1) = rotationMatrix2(1,:)*vLocalSec';
                                            albedo(prod2).v(2) = rotationMatrix2(2,:)*vLocalSec';
                                            albedo(prod2).v(3) = rotationMatrix2(3,:)*vLocalSec';
                                            clear vLocalSec
                                            rLocalSec = [albedo(prod2).r(1) albedo(prod2).r(2) albedo(prod2).r(3)];
                                            albedo(prod2).r(1) = rotationMatrix2(1,:)*rLocalSec';
                                            albedo(prod2).r(2) = rotationMatrix2(2,:)*rLocalSec';
                                            albedo(prod2).r(3) = rotationMatrix2(3,:)*rLocalSec';
                                            clear rLocalSec
                                        end

                                        % Secondaries (albedo) loop
                                        if Gen+1 < GenMax
                                            for prod2 = 1:length(albedo)
                                                if verb > 2
                                                    fprintf('\n%s%s%2u%s%2u%s%2u%s%s%10s%s%5.3f%s', ...
                                                        repmat('   ', 1, Gen+3), 'Gen: ', Gen+2, '  (', prod2, ' of ', length(albedo), ')  ', ' = ', ...
                                                        albedo(prod2).ParticleName, ' (', albedo(prod2).E, ' GeV) = ');
                                                end

                                                % Coordinates of secondaries in XYZ in m
                                                R2 = xyz2cen + albedo(prod2).r*1e3;

                                                RunGetTrajectory(ProductTrace, TraceData, albedo, prod2, EnMin, Exclude, R2./Rsconv, Rs, CRF, Date, varargin, Tstart, TotTime, Gen+1, verb);
                                            end
                                        end                                        
                                    else
                                        if verb >= 2
                                            fprintf('%s%16s%s', '   ~ ', 'skip', ' ~ ');
                                        end
                                    end
                                else
                                    error('   GT: can''''t run EAS simulation')
                                end
                            end
                        end
                    end
                end

                [LocalPathDen, LocalDen, LocalChemComp, nLocal] = deal(0);
                [LocalDenAlong, LocalCoord] = deal([]);
                clear rLocal vLocal Tafter status process product Vcorr
                clear rotationVector rotationMatrix rotationMatrix2
            end % If PathDen > critical
        end % If inelastic interactions

        % ----------------------------------------------
        % ------------------------    Save current point
        % ----------------------------------------------
        kl = 0;
        if Nsave > 0
            if mod(istep, Nsave) == 0
                k = k + 1;
                Xs(k) = X; 
                Ys(k) = Y; 
                Zs(k) = Z;
                if MFFsave == 1
                    Hs(k-1,:) = [Hx Hy Hz];
                end
                if EFFsave == 1
                    Es(k-1,:) = [Ex Ey Ez];
                end
                if Tsave == 1
                    Ts(k) = Tpart;
                end
                if Asave == 1
                    As(k-1) = atan2(norm(cross(XYZPrev,XYZCur)), dot(XYZPrev,XYZCur));
                    XYZPrev = XYZCur;
                end
                if Lsave == 1
                    Ls(k-1,:) = [PathLen PathDen];
                end
                kl = 1;
            end
        end

        % ----------------------------------------------
        % -------------------------      Full revolution
        % ----------------------------------------------
        if (GTparam.GTmag.ParticleOriginIsOn || GTparam.BC.MaxRevIsOn) && UseEMfield
            [a_, b_, ~] = geo2mag_eccentric_gh(XYZCur(1), XYZCur(2), XYZCur(3), 1, IGRF.g, IGRF.h);
        
            lon = atan(b_ / a_);
            lon_tolerance = deg2rad(60);
        
            if istep == 1
                lon_prev = lon;
                lon_total = 0;
            end    
        
            lon_diff = abs(lon - lon_prev);
            lon_diff(lon_diff > pi/2) = pi - lon_diff;
        
            if lon_diff > lon_tolerance
                lon_total = lon_total + lon_diff;
                lon_prev = lon;
            end
            
            lon_total = abs(lon_total);
            
            if abs(lon_total) > 2*pi
                full_revolutions = full_revolutions + 1;
                lon_total = rem(lon_total, 2*pi);
            end
        end

        % ----------------------------------------------
        % -------------------------     Break Conditions
        % ----------------------------------------------

        RDistance = norm(XYZCur);
        Dist2Path = sqrt( (X-Xs(1))^2 + (Y-Ys(1))^2 + (Z-Zs(1))^2 ) / TotPathLen;

        if BCProps.UseDefault
            DBCcur = [abs(X) abs(Y) abs(Z) RDistance Dist2Path];
            ShouldBreakDefault(1:5) = abs(DBCcur) < DBC(1:5);
            DBCcur = [abs(X) abs(Y) abs(Z) RDistance TotPathLen TotPathDen TotTime];
            ShouldBreakDefault(6:12) = abs(DBCcur) > DBC(6:12);
            ShouldBreakDefault(13) = full_revolutions >= DBC(13);
        end

        if BCProps.UseFunction
            StepData.XYZCur = XYZCur;
            StepData.R = RDistance;
            StepData.XYZPrev = XYZPrev; 

            StepData.TotalPath = TotPathLen;
            StepData.PathDen = PathDen;
            StepData.Dist2Path = Dist2Path;

            StepData.TimeStep = dt;
            StepData.TotalTime = TotTime;

            StepData.VXYZ = [Vxm Vym Vzm];
            StepData.VMag = norm(StepData.VXYZ);

            StepData.HXYZ = [Hx Hy Hz];
            StepData.HMag = norm(StepData.HXYZ);

            StepData.EXYZ = [Ex Ey Ez];
            StepData.EMag = norm(StepData.EXYZ);

            StepData.LengthScale = Rsconv;
            StepData.KinEnergy = Tpart;
            
            StepData.FullRev = full_revolutions;            
            StepData.UserParam = UserParam;            
            StepData.IGRF = IGRF;
            StepData.Dates = Dates;
            StepData.istep = istep;
            
            [ShouldBreakUserFunction, UserParam] = BCProps.UserBCFunction(StepData);
        end

        if any(ShouldBreakDefault) || ShouldBreakUserFunction
            WOut = 1;
            if any(ShouldBreakDefault)
                DefaultBCExitCause = char(strjoin(char(BCProps.DefaultBCNames(ShouldBreakDefault)), '_'));
                Status = ['DefaultBC_' DefaultBCExitCause];
            elseif ShouldBreakUserFunction
                UserBCExitCause = func2str(BCProps.UserBCFunction);
                Status = 'UserBCFunction';
            end
            break
        end

        if IsPrimInteract == 1
            WOut = 2;
            Status = 'ParticleInteract';
            break
        end

        if IsPrimDecay == 1
            WOut = 3;
            Status = 'ParticleDecay';
            break
        end
    end % Main loop
    %toc
    
    % ----------------------------------
    % >>> Final setup (part 1 of 2)
    % ----------------------------------
    % Save the last point
    if kl == 0
        k=k+1;
        Xs(k) = X; 
        Ys(k) = Y;
        Zs(k) = Z;
        if Asave == 1
            As(k-1) = atan2(norm(cross(XYZPrev,XYZCur)), dot(XYZPrev,XYZCur)); 
        end
        if Lsave == 1
            Ls(k-1,:) = [PathLen PathDen];
        end
    end
    Ts(find(Ts > 0, 1, 'last') + 1) = Tpart;

    Xs(k+1:end)   = [];
    Ys(k+1:end)   = [];
    Zs(k+1:end)   = [];
    Hs(k:end,:) = [];
    Es(k:end,:) = [];
    Ts(k+1:end)   = [];
    As(k:end)   = [];
    Ls(k:end,:) = [];

    Ts = Ts/kes;                                % MeV -> input scale
    
    TotPathLen = TotPathLen/Rsconv;
    TrackInfo = [TotTime TotPathLen TotPathDen];

    % ----------------------------------
    % >>> Particles in magnitosphere (1)
    % ----------------------------------
    if GTparam.GTmag.TrackParamIsOn && UseEMfield
        if verb > 1
            fprintf('\n   Get Trajectory parameters (GTmag) ... ')
        end
        GTtrack.data.GTmag = GetGTtrackParam([Xs Ys Zs], Hs, GTparticle, GTparam);
        if GTparam.GTmag.IsFirstRun
            parReq  = GTtrack.data.GTmag.GuidingCentre.parReq;
            Req     = GTtrack.data.GTmag.GuidingCentre.Req;
            Rline   = GTtrack.data.GTmag.GuidingCentre.Rline;
        else
            [parReq, Req, Rline] = deal([]);
        end
        if verb > 1
            fprintf(' done \n')
        end
    end
    
    % ----------------------------------
    % >>> Final setup (part 2 of 2)
    % ----------------------------------
    if ~isempty(CRFout) && ~strcmp(CRFout, 'geo')
        switch lower(CRFout)
            case 'dipmag'
                [Ro(1), Ro(2), Ro(3)] = geo2dipmag(Ro(1), Ro(2), Ro(3), GTparam.EarthBfield.GTB.DIP.psi, -1);
                [Vo(1), Vo(2), Vo(3)] = geo2dipmag(Vo(1), Vo(2), Vo(3), GTparam.EarthBfield.GTB.DIP.psi, -1);
            case 'mag'
                [Xs, Ys, Zs] = geo2mag_eccentric(Xs, Ys, Zs, 1, Dates.DTnum);
                if ~isempty(Rint)
                    [Rint(1), Rint(2), Rint(3)] = geo2mag_eccentric(Rint(1), Rint(2), Rint(3), 1, Dates.DTnum);
                end
                if GTparam.GTmag.TrackParamIsOn
                    [parReq(:,1), parReq(:,2), parReq(:,3)] = geo2mag_eccentric(parReq(:,1), parReq(:,2), parReq(:,3), 1, Dates.DTnum);
                    [Req(:,1), Req(:,2), Req(:,3)] = geo2mag_eccentric(Req(:,1), Req(:,2), Req(:,3), 1, Dates.DTnum);
                    [Rline(:,1), Rline(:,2), Rline(:,3)] = geo2mag_eccentric(Rline(:,1), Rline(:,2), Rline(:,3), 1, Dates.DTnum);
                end
            case 'gsm'                
                [Xs, Ys, Zs] = geo2gsm(Xs, Ys, Zs, Dates.Year, Dates.DoY, Dates.Secs, 1);
                if ~isempty(Rint)
                    [Rint(1), Rint(2), Rint(3)] = geo2gsm(Rint(1), Rint(2), Rint(3), Dates.Year, Dates.DoY, Dates.Secs, 1);
                end
                if GTparam.GTmag.TrackParamIsOn
                    [parReq(:,1), parReq(:,2), parReq(:,3)] = geo2gsm(parReq(:,1), parReq(:,2), parReq(:,3), Dates.Year, Dates.DoY, Dates.Secs, 1);
                    [Req(:,1), Req(:,2), Req(:,3)] = geo2gsm(Req(:,1), Req(:,2), Req(:,3), Dates.Year, Dates.DoY, Dates.Secs, 1);
                    [Rline(:,1), Rline(:,2), Rline(:,3)] = geo2gsm(Rline(:,1), Rline(:,2), Rline(:,3), Dates.Year, Dates.DoY, Dates.Secs, 1);
                end
        end
    end

    Xs = Xs/Rsconv;                             % m -> input scale
    Ys = Ys/Rsconv;
    Zs = Zs/Rsconv;    
    if ~isempty(Ls)
        Ls(:,1) = Ls(:,1)/Rsconv;
    end

    if GTparam.GTmag.TrackParamIsOn && UseEMfield
        GTtrack.data.GTmag.GuidingCentre.parReq = parReq/GTparam.Ro.Rsconv;
        GTtrack.data.GTmag.GuidingCentre.Req = Req/GTparam.Ro.Rsconv;
        GTtrack.data.GTmag.GuidingCentre.Rline = Rline/GTparam.Ro.Rsconv;
    end

    % ----------------------------------
    % >>> Save GTTrack
    % ----------------------------------
    GTtrack.data.Track.X  = Xs;
    GTtrack.data.Track.Y  = Ys;
    GTtrack.data.Track.Z  = Zs;
    GTtrack.data.Track.T  = Ts;
    GTtrack.data.Track.dt = dt;
    if Csave
        GTtrack.data.Track.t = (Tstart:dt:Tstart + TotTime + dt/2)'; % dt/2 correction for precision
    end
    GTtrack.data.Track.Rint = Rint;
    if MFFsave
        GTtrack.data.Fields.H = Hs;
    end
    if EFFsave
        GTtrack.data.Fields.E = Es;
    end
    if Asave
        GTtrack.data.Track.A = As;
    end
    if Lsave
        GTtrack.data.Track.L = Ls;
    end
    GTtrack.data.TrackInfo = TrackInfo;
    GTtrack.data.Gen  = Gen;
    GTtrack.data.BC.WOut = WOut;
    GTtrack.data.GTparam = GTparam;
    GTtrack.data.GTparticle = GTparticle;
    if WOut == 0
         Status = 'done';
    end
    GTtrack.data.BC.Status = Status;
    GTtrack.data.BC.lon_total = lon_total;
    GTtrack.data.BC.UserParam = UserParam;
    
    % ----------------------------------
    % >>> Particles in magnitosphere (2)
    % ----------------------------------
    if GTparam.GTmag.ParticleOriginIsOn && GTparam.GTmag.IsFirstRun && UseEMfield
        if verb > 1
            fprintf('   Get particle origin ... ')
        end
        GTtrack.data.GTmag.ParticleOrigin = FindParticleOrigin(GTtrack);
        if verb > 1
            fprintf([GTtrack.data.GTmag.ParticleOrigin.Name ' (' num2str(GTtrack.data.GTmag.ParticleOrigin.Ind) ')'])
        end
    end
    
    % ----------------------------------
    % >>> Final Verbose
    % ----------------------------------
    if verb > 1
        fprintf('\n   - \n   Done! \n   - \n')
        if TotTime < 1
            fprintf(['   Total time: ' num2str(TrackInfo(1)) ' s \n'])
        else
            DS = datetime(0,01,01,0,0,TrackInfo(1));
            fprintf(['   Total time: \n' ...
                                    '      ', num2str(year(DS)) ' years, ' ...
                                              num2str(month(DS)-1) ' montsh, ' ...
                                              num2str(day(DS)-1) ' days, \n' ...
                                    '      ', num2str(hour(DS)) ' hours, ' ...
                                              num2str(minute(DS)) ' minutes, ' ...
                                              num2str(second(DS)) ' seconds, \n ' ...
                                    '     or ' num2str(TrackInfo(1)) ' s \n'])
        end
        if UseMedium == 1
            fprintf(['   Total path: ' num2str(TrackInfo(2)) ' ' Rs ' or ' num2str(TrackInfo(3)) ' g/cm2 \n'])
        else
            fprintf(['   Total path: ' num2str(TrackInfo(2)) ' ' Rs ' \n'])
        end
        if WOut == 1
            fprintf(['\n   == Exit due to break condition (' DefaultBCExitCause UserBCExitCause ') == \n'])
        elseif WOut == 2
            fprintf('\n   == Exit due to particle interaction == \n')
        elseif WOut == 3
            fprintf('\n   == Exit due to particle decay == \n')
        elseif WOut == 4
            fprintf('\n   == Exit due to fly out from EM field region == \n')
        end
    end
end

%% ==================================================================================================
%  ----- >>>>> Functions
%  ==================================================================================================

function [Hx, Hy, Hz] = GetGTBfield(RegionCode, GTB, Units, Dates, X, Y, Z)
    Hx = 0; 
    Hy = 0; 
    Hz = 0;

    % Magnetosphere
    if RegionCode == 1                    
        if GTB.GetD == 1
            %tic
            [Bxd, Byd, Bzd] = GetDipoleB(GTB.DIP.psi, GTB.DIP.magmom, ...
                                         X/Units.RE2m, Y/Units.RE2m, Z/Units.RE2m);
            Hx = Hx + Bxd;
            Hy = Hy + Byd;
            Hz = Hz + Bzd;
            %toc
        end
        if GTB.GetI == 1
            %tic
            %[Bxi, Byi, Bzi] = GetIGRFB(X*1e-3, Y*1e-3, Z*1e-3, Dates.DTnum);
            [Bxi, Byi, Bzi] = GetIGRFB_gh(X*1e-3, Y*1e-3, Z*1e-3, GTB.IGRF.gh);
            Hx = Hx + Bxi;
            Hy = Hy + Byi;
            Hz = Hz + Bzi;
            %toc
        end
        if GTB.GetC == 1
            %tic
            %[Bxc, Byc, Bzc] = GetCHAOSB(X*1e-3, Y*1e-3, Z*1e-3, Dates.Date, ...
            %                            GTB.CHAOS.IsIntCore, GTB.CHAOS.IsIntCrustal, GTB.CHAOS.IsExt);
            [Bxc, Byc, Bzc] = GetCHAOSB(X*1e-3, Y*1e-3, Z*1e-3, GTB.CHAOS);
            Hx = Hx + Bxc;
            Hy = Hy + Byc;
            Hz = Hz + Bzc;
            %toc
        end
        if GTB.GetT == 1
            %tic                 
            [Bxt, Byt, Bzt] = GetTsyganenkoB(X/Units.RE2m, Y/Units.RE2m, Z/Units.RE2m, ...
                                        GTB.TSYG.psit, GTB.TSYG.ind, Dates.Year, Dates.DoY, Dates.Secs, GTB.TSYG.ModCode);
            Hx = Hx + Bxt;
            Hy = Hy + Byt;
            Hz = Hz + Bzt;
            %toc
        end

    % Heliosphere
    elseif RegionCode == 2
        if GTB.GetH == 1
            T0 = GTB.T0; 
            [Hx, Hy, Hz] = GetBField(T0, X/Units.AU2m, Y/Units.AU2m, ...,
                                     Z/Units.AU2m, 'pol', GTB.pol, 'cir', ...
                                     GTB.cir, 'noise', GTB.noise);
        end
        
    % Galaxy
    elseif RegionCode == 3
        if GTB.GetG == 1
            [Hx, Hy, Hz] = JF12mod(X/Units.kpc2m, Y/Units.kpc2m, Z/Units.kpc2m);
        end

    % Pulsar
    elseif RegionCode == 4
        [Hx, Hy, Hz] = GetDipoleB(GTB.DIP.psi, GTB.DIP.magmom, ...
                                  X/1e3, Y/1e3, Z/1e3);

    end

    Hx = Hx * 1e-9;     % nT -> T
    Hy = Hy * 1e-9;
    Hz = Hz * 1e-9;
end

function Trace = RunGetTrajectory(Trace, TraceData, product, prod, EnMin, Exclude, Rint, Rs, CRF, Date, vars, Tstart, TotTime, Gen, verb)
    if product(prod).E > EnMin && ~ismember(product(prod).PDG(1), Exclude)
        if ~isempty(CRF)
            CRF{1} = 'geo';
        end
        Trace.attach_child(GetTrajectoryInEMField({[Rint(1) Rint(2) Rint(3)], Rs, CRF}, ...
                                    [product(prod).v(1) product(prod).v(2) product(prod).v(3)], ...
                                    {product(prod).PDG, product(prod).E, 'GeV'}, ...
                                    Date, ...                       % Date
                                    vars{5}, ...                    % Region
                                    vars{6}, ...                    % EMFF
                                    vars{7}, ...                    % Steps
                                    vars{8}, ...                    % Medium
                                    {vars{9}{1:end}, {Tstart+TotTime Gen+1}}, ...  % Interaction
                                    vars{10}, ...                   % Save mode
                                    vars{11}, ...                   % Break conditions
                                    2*(verb > 2) + 0) ...           % Verbose 2*(verb > 2) + 0
        );
    else
        FillTrace(Trace, TraceData, product(prod).PDG(1), Gen+1, 'skip', ExitCause.Skipped);
        if verb > 2
            fprintf('%s%16s%s', '   ~ ', 'skip', ' ~ ');
        end
    end
end

function NewChild = FillTrace(Trace, TraceData, PDG, Gen, P, ExitCauseArg)
    TraceEmpty = TNode(TraceData);
    TraceEmpty.data.PDG = int64(PDG);
    TraceEmpty.data.Gen = Gen;
    TraceEmpty.data.WOut = P;
    TraceEmpty.data.ExitCause = ExitCauseArg;
    [~, NewChild] = Trace.attach_child(TraceEmpty);
end


