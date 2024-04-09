function [r , v, Energy, ParticleName] = GeneratorCR(Source, Spectrum, Particle, Nevents, Verbose)
%   [r , v, Energy, ParticleName] = GeneratorCR(arguments)
% 	Simulation of Cosmic Ray spectra and chemical composition
%   Ver. 2, red. 1 / June 2023 / R. Yulbarisov, A. Mayorov / CRTeam / NRNU MEPhI, Russia
%
%   The function generates cosmic ray particles in two modes.
%   1) Inward. Particles are uniformly distributed on a sphere with a given radius. The direction of
%      the initial velocity of the particles is played out so that there is an isotropic flux at
%      any point inside the sphere.
%   2) Outward. Particles are generated at a point with the specified coordinates. The direction of
%      the initial velocity of the particles is played out isotropically.
%   The particle energy or rigidity distribution is described by different types of spectrum.
%
%   Arguments:
%       Source      -   Contains parameters of the sphere:
%                           Mode        -   generation mode: 'inward' or 'outward'
%                           Radius      -   radius of the sphere
%                           Center      -   coordinates of the center of the sphere or of the point
%                       The format, valid input and default values are described in the GetSourceArguments function.
%       Spectrum    -   Contains parameters of the energy distribution of particles:
%                           T, Tmin, Tmax and similar parameters    -   boundaries of the energy distribution in [GeV/nuc] or [GV] or [GeV],
%                           Base        -   type of spectrum being played:
%                                             '' for mono line
%                                             'T' or 'R' or 'E' for power spectrum with corresponding base
%                                             'F' for spectrum according to Force-field model
%                           Index       -   spectrum parameter:
%                                             power index for power spectrum
%                                             solar modulation potential in [GV] for Force-field spectrum
%                       The format, valid input and default values are described in the GetSpectrumArguments function.
%       Particle    -   Contains parameters of the chemical composition of particles:
%                           Type        -   types of played particles, to reproduce the natural abundance of particles in GCR, use {'GCR'}
%                           Abundance   -   abundance of particles, this parameter is ignored for the case {'GCR'}
%                       The format, valid input and default values are described in the GetParticleArguments function.
%       Nevents     -   Number of events played.
%       Verbose     -   Verbose (0 or 1).
%
%   Output:
%       r               -   Coordinates of particles, same scale as input.
%       v               -   Velocity of particles (normalized), same scale as input.
%       Energy          -   Energy of particles according to spectrum.
%       ParticleName    -   Name of particles.
%
%   Examples:
%       [r, v, Energy, ParticleName] = GeneratorCR({'Radius', 1, 'Center', [1 2 3]}, {'R', 5}, {'Type', {'pr', 'he4'}, 'Abundance', [0.9, 0.1]}, 1e2);
%       [r, v, Energy, ParticleName] = GeneratorCR({'Radius', 5}, {'Tmin', 1, 'Tmax', 20, 'Base', 'R', 'Index', -2.7}, {'Type', {'he4'}}, 1e4);
%       [r, v, Energy, ParticleName] = GeneratorCR({'Radius', 1}, {'Rmin', 1, 'Rmax', 20, 'Base', 'F', 'Index', 0.9}, {'Type', {'pr', 'he4'}, 'Abundance', [0.9, 0.1]}, 1e5);
%       [r, v, Energy, ParticleName] = GeneratorCR({'Radius', 1}, {'Emin', 1, 'Emax', 20, 'Base', 'F', 'Index', 1.2}, {'Type', {'GCR'}}, 1e5);
%       [r, v, Energy, ParticleName] = GeneratorCR({'Mode', 'outward', 'Center', [1 2 3]}, {'Emin', 1, 'Emax', 20, 'Base', 'F', 'Index', 0.6}, {'Type', {'GCR'}}, 1e5);

    arguments
        Source      (1,:)   cell
        Spectrum    (1,:)   cell
        Particle    (1,:)   cell
        Nevents     (1,1)   double  {mustBeInteger}                 = 1
        Verbose     (1,1)   double  {mustBeMember(Verbose, [0, 1])} = 0
    end

    % Read input
    if Verbose == 1
        fprintf('   Reading input ...\n')
    end
    [Mode, Ro, Rc] = GetSourceArguments(Source{:});
    [EnergyRangeUnits, EnergyRange, SpectrumBase, SpectrumIndex] = GetSpectrumArguments(Spectrum{:});
    [Nparticles, ParticleType, Abundance] = GetParticleArguments(Particle{:});
    if Nparticles > Nevents
        error('    Number of particles > Number of events')
    end
    if Verbose == 1
        fprintf('      + Source parameters\n')
        fprintf('        Ro = %d\n', Ro)
        fprintf('        Rc = [%.1f, %.1f, %.1f]\n', Rc)
        fprintf('      + Spectrum\n')
        if length(EnergyRange) == 1
            fprintf('        EnergyRange = %.1f\n', EnergyRange)
        else
            fprintf('        EnergyRange = [%.1f, %.1f]\n', EnergyRange)
        end
        fprintf('        EnergyRangeUnits = %s\n', EnergyRangeUnits)
        fprintf('        SpectrumBase = %s\n', SpectrumBase)
        fprintf('        SpectrumIndex = %.1f\n', SpectrumIndex)
        if Nparticles
            fprintf('      + Chemical composition: %d particles\n', Nparticles)
            for s = 1:Nparticles
                fprintf('           Particle: %s\n', ParticleType{s})
                fprintf('           Abundance: %.1f\n', Abundance(s))
            end
        else
            fprintf('      + Chemical composition: GCR\n')
        end
        fprintf('   Number of events: %d\n', Nevents)
    end

    % Simulation
    if Verbose == 1
        fprintf('   Simulation ...\n')
        fprintf('      + Coordinates and velocities\n')
    end
    switch Mode
    case 'inward'
        theta = acos(1 - 2 * rand(Nevents, 1));
        phi = 2 * pi * rand(Nevents, 1);
        r = [sin(theta) .* cos(phi), ...
             sin(theta) .* sin(phi), ...
             cos(theta)];
        newZ = permute(r, [2 3 1]);
        newX = cross(newZ, repmat([0; 0; 1], 1, 1, Nevents));
        newY = cross(newZ, newX);
        S = [newX ./ vecnorm(newX), ...
             newY ./ vecnorm(newY), ...
             newZ ./ vecnorm(newZ)];
        ksi = rand(1, 1, Nevents);
        sin_theta = sqrt(ksi);
        cos_theta = sqrt(1 - ksi);
        phi = 2 * pi * rand(1, 1, Nevents);
        p = [-sin_theta .* cos(phi); ...
             -sin_theta .* sin(phi); ...
             -cos_theta];
        r = r * Ro + Rc;
        v = pagemtimes(S, p);
        v = permute(v, [3 1 2]);
    case 'outward'
        r = repmat(Rc, Nevents, 1);
        theta = acos(1 - 2 * rand(Nevents, 1));
        phi = 2 * pi * rand(Nevents, 1);
        v = [sin(theta) .* cos(phi), ...
             sin(theta) .* sin(phi), ...
             cos(theta)];
    end

    if Verbose == 1
        fprintf('      + Energy spectra\n')
    end
    if Nparticles
        % User's chemical composition case
        Energy = zeros(Nevents, 1);
        Nabundance = floor(Nevents * Abundance);
        if sum(Nabundance) ~= Nevents
            Nabundance(end) = Nevents - sum(Nabundance(1:end-1));
        end
        Nindex = [0, cumsum(Nabundance)];
        ParticleName = repelem(ParticleType, Nabundance)';
        for s = 1:Nparticles
            i1 = Nindex(s) + 1;
            i2 = Nindex(s+1);
            switch SpectrumBase
            case ''
                % ------- Mono Energy -------
                Energy(i1:i2) = EnergyRange;
            case {'T', 'R', 'E'}
                % ------- Power Spectrum -------
                [A, Z, M, ~, ~, ~] = GetNucleiProp(ParticleType{s});
                M = M / 1e3; % MeV/c2 -> GeV/c2
                if ~strcmp(EnergyRangeUnits, SpectrumBase)
                    EnergyRangeS = ConvertUnits(EnergyRange, EnergyRangeUnits, SpectrumBase, M, A, Z);
                else
                    EnergyRangeS = EnergyRange;
                end
                ksi = rand(Nabundance(s), 1);
                if SpectrumIndex == -1
                    Energy(i1:i2) = EnergyRangeS(1) * (EnergyRangeS(2) / EnergyRangeS(1)).^ksi;
                else
                    g = SpectrumIndex + 1;
                    Energy(i1:i2) = (EnergyRangeS(1)^g + ksi * (EnergyRangeS(2)^g - EnergyRangeS(1)^g)).^(1 / g);
                end
                if ~strcmp(EnergyRangeUnits, SpectrumBase)
                    Energy(i1:i2) = ConvertUnits(Energy(i1:i2), SpectrumBase, EnergyRangeUnits, M, A, Z);
                end
            case 'F'
                % ------- Force-field -------
                [A, Z, M, ~, ~, ~] = GetNucleiProp(ParticleType{s});
                M = M / 1e3; % MeV/c2 -> GeV/c2
                if ~strcmp(EnergyRangeUnits, 'T')
                    EnergyRangeS = ConvertUnits(EnergyRange, EnergyRangeUnits, 'T', M, A, Z);
                else
                    EnergyRangeS = EnergyRange;
                end
                Jmax = max(GetGCRflux('T', logspace(log10(EnergyRangeS(1)), log10(EnergyRangeS(2)), 1e3), ParticleType{s}, SpectrumIndex));
                iFilled = 0;
                bunchSize = 1e6; % for faster computing
                while true
                    Eplayed = EnergyRangeS(1) + rand(bunchSize, 1) * (EnergyRangeS(2) - EnergyRangeS(1));
                    ksi = Jmax * rand(bunchSize, 1);
                    Esuited = Eplayed(ksi < GetGCRflux('T', Eplayed, ParticleType{s}, SpectrumIndex));
                    if length(Esuited) < Nabundance(s) - iFilled
                        Energy(i1+iFilled:i1+iFilled+length(Esuited)-1) = Esuited;
                        iFilled = iFilled + length(Esuited);
                    else
                        iCut = Nabundance(s) - iFilled;
                        Energy(i1+iFilled:i2) = Esuited(1:iCut);
                        break;
                    end
                end
                if ~strcmp(EnergyRangeUnits, 'T')
                    Energy(i1:i2) = ConvertUnits(Energy(i1:i2), 'T', EnergyRangeUnits, M, A, Z);
                end
            end
        end
    else
        % GCR chemical composition case
        ParticleType = {'pr', 'he4', 'Li-7', 'Be-9', 'B-11', 'C-12', 'N-14', 'O-16', 'F-19', 'Ne-20', ...
                        'Na-23', 'Mg-24', 'Al-27', 'Si-28', 'Fe-56', 'ele', 'pos', 'apr'};
        EnergyArgument = {[EnergyRangeUnits, 'min'], EnergyRange(1), [EnergyRangeUnits, 'max'], EnergyRange(2), 'Base', 'F', 'Index', SpectrumIndex};
        if Nevents > 1
            [~, ~, Energy, ~] = GeneratorCR({'Radius', 1}, EnergyArgument, {'Type', {'pr', 'he4'}, 'Abundance', [0.9, 0.1]}, Nevents, 0);
        else
            if rand < 0.9
                [~, ~, Energy, ~] = GeneratorCR({'Radius', 1}, EnergyArgument, {'Type', {'pr'}}, Nevents, 0);
            else
                [~, ~, Energy, ~] = GeneratorCR({'Radius', 1}, EnergyArgument, {'Type', {'he4'}}, Nevents, 0);
            end
        end
        Jparticle = GetGCRflux(EnergyRangeUnits, Energy, '', SpectrumIndex);
        iParticle = sum(rand(Nevents, 1) > cumsum(Jparticle ./ sum(Jparticle, 2), 2), 2) + 1;
        ParticleName = ParticleType(iParticle)';
    end
end

function [Mode, Ro, Rc] = GetSourceArguments(SourceArguments)
    arguments
        SourceArguments.Mode            char    {mustBeMember(SourceArguments.Mode, ...
                                                 {'inward', 'outward'})}                    = 'inward'
        SourceArguments.Radius  (1,1)   double  {mustBeNonnegative}                         = 1
        SourceArguments.Center  (1,3)   double                                              = [0 0 0]
    end
    Mode = SourceArguments.Mode;
    Ro = SourceArguments.Radius;
    Rc = SourceArguments.Center;
    if strcmp(Mode, 'inward')
        mustBePositive(Ro);
    end
end

function [EnergyRangeUnits, EnergyRange, SpectrumBase, SpectrumIndex] = GetSpectrumArguments(SpectrumArguments)
    arguments
        SpectrumArguments.T     (1,1)   double  {mustBeNonnegative}                         = 0
        SpectrumArguments.Tmin  (1,1)   double  {mustBeNonnegative}                         = 0
        SpectrumArguments.Tmax  (1,1)   double  {mustBeNonnegative}                         = inf
        SpectrumArguments.R     (1,1)   double  {mustBeNonnegative}                         = 0
        SpectrumArguments.Rmin  (1,1)   double  {mustBeNonnegative}                         = 0
        SpectrumArguments.Rmax  (1,1)   double  {mustBeNonnegative}                         = inf
        SpectrumArguments.E     (1,1)   double  {mustBeNonnegative}                         = 0
        SpectrumArguments.Emin  (1,1)   double  {mustBeNonnegative}                         = 0
        SpectrumArguments.Emax  (1,1)   double  {mustBeNonnegative}                         = inf
        SpectrumArguments.Base          char    {mustBeMember(SpectrumArguments.Base, ...
                                                 {'', 'T', 'R', 'E', 'F'})}                 = ''
        SpectrumArguments.Index (1,1)   double                                              = 1
    end

    caseVector = [SpectrumArguments.T > 0, SpectrumArguments.Tmin > 0 & SpectrumArguments.Tmax < inf, ...
                  SpectrumArguments.R > 0, SpectrumArguments.Rmin > 0 & SpectrumArguments.Rmax < inf, ...
                  SpectrumArguments.E > 0, SpectrumArguments.Emin > 0 & SpectrumArguments.Emax < inf];
    if sum(caseVector) ~= 1
        error('    The particle energy is incorrectly set')
    end
    switch find(caseVector)
    case 1
        EnergyRangeUnits = 'T';
        EnergyRange = SpectrumArguments.T;
    case 2
        EnergyRangeUnits = 'T';
        EnergyRange = [SpectrumArguments.Tmin, SpectrumArguments.Tmax];
        mustBeGreaterThanOrEqual(SpectrumArguments.Tmax, SpectrumArguments.Tmin)
    case 3
        EnergyRangeUnits = 'R';
        EnergyRange = SpectrumArguments.R;
    case 4
        EnergyRangeUnits = 'R';
        EnergyRange = [SpectrumArguments.Rmin, SpectrumArguments.Rmax];
        mustBeGreaterThanOrEqual(SpectrumArguments.Rmax, SpectrumArguments.Rmin)
    case 5
        EnergyRangeUnits = 'E';
        EnergyRange = SpectrumArguments.E;
    case 6
        EnergyRangeUnits = 'E';
        EnergyRange = [SpectrumArguments.Emin, SpectrumArguments.Emax];
        mustBeGreaterThanOrEqual(SpectrumArguments.Emax, SpectrumArguments.Emin)
    end

    SpectrumBase = SpectrumArguments.Base;
    if length(EnergyRange) == 1 && strcmp(SpectrumBase, 'F')
        error('    Force-field simulation for mono line can not be done')
    end

    SpectrumIndex = SpectrumArguments.Index;
    if strcmp(SpectrumBase, 'F') && SpectrumIndex < 0
        error('    Modulation potential should be positive')
    end
    % !!! !!! SpectrumIndex < 0.2 temporary crutch - remove after updating GetGCRGlux !!! !!!
    if strcmp(SpectrumBase, 'F') && SpectrumIndex < 0.2
        error('    Modulation potential < 0.2 GV work wrong')
    end
    % !!! !!! SpectrumIndex < 0.2 temporary crutch - remove after updating GetGCRGlux !!! !!!
end

function [Nparticles, ParticleType, Abundance] = GetParticleArguments(ParticleArguments)
    arguments
        ParticleArguments.Type      (1,:)   cell    = {'pr'}
        ParticleArguments.Abundance (1,:)   double  = 1
    end
    Nparticles = length(ParticleArguments.Type);

    ParticleType = ParticleArguments.Type;
    if Nparticles == 1 && strcmp(ParticleType{1}, 'GCR')
        Nparticles = 0;
    end

    Abundance = ParticleArguments.Abundance;
    if abs(sum(Abundance) - 1) > 1e-16
        error('    sum(Abundance) should be equal to 1')
    end
end

function EnergyConverted = ConvertUnits(Energy, FromUnits, ToUnits, M, A, Z)
    switch FromUnits
    case 'T'
        switch ToUnits
        case 'R'
            EnergyConverted = ConvertT2R(Energy, M, A, Z);
        case 'E'
            EnergyConverted = Energy * A;
        end
    case 'R'
        switch ToUnits
        case 'T'
            EnergyConverted = ConvertR2T(Energy, M, A, Z);
        case 'E'
            EnergyConverted = ConvertR2T(Energy, M, A, Z) * A;
        end
    case 'E'
        switch ToUnits
        case 'T'
            EnergyConverted = Energy / A;
        case 'R'
            EnergyConverted = ConvertT2R(Energy / A, M, A, Z);
        end
    end
end
