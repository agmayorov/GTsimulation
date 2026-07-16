function Model = GetBfieldFullModelName(MNparam)
%   Function to get full name of a planetary MF Model
%   Ver. 1, red. 3 / 28 June 2023 / A. Mayorov
%
%   Possible cases:
%   IGRF
%       Model = GetBfieldFullModelName('Planet', 'Earth', 'Model', 'IGRF', 'Ver', 13, 'Type', 'core');
%       Model = GetBfieldFullModelName('Planet', 'Earth', 'Model', 'IGRF', 'Ver', 13, 'Type', 'core', 'Target', 'TxtFileLoc');
%   CHAOS
%       Model = GetBfieldFullModelName('Planet', 'Earth', 'Model', 'CHAOS', 'Ver', 7.13, 'Type', 'core');
%       Model = GetBfieldFullModelName('Planet', 'Earth', 'Model', 'CHAOS', 'Ver', 7.13, 'Type', 'static');
%   CM
%       Model = GetBfieldFullModelName('Planet', 'Earth', 'Model', 'CM', 'Ver', 6, 'Type', 'core');
%       Model = GetBfieldFullModelName('Planet', 'Earth', 'Model', 'CM', 'Ver', 6, 'Type', 'static');
%       Model = GetBfieldFullModelName('Planet', 'Earth', 'Model', 'CM', 'Ver', 6, 'Type', 'ionosphere');
%   COV-OBS
%       Model = GetBfieldFullModelName('Planet', 'Earth', 'Model', 'COV-OBS', 'Ver', 2, 'Type', 'core');
%   LCS
%       Model = GetBfieldFullModelName('Planet', 'Earth', 'Model', 'LCS', 'Ver', 1, 'Type', 'static');
%   DIFI
%       Model = GetBfieldFullModelName('Planet', 'Earth', 'Model', 'DIFI', 'Ver', 6, 'Type', 'ionosphere');
%   SIFM
%       Model = GetBfieldFullModelName('Planet', 'Earth', 'Model', 'SIFM', 'Type', 'core');
%       Model = GetBfieldFullModelName('Planet', 'Earth', 'Model', 'SIFM', 'Type', 'static');
%   JRM33
%       Model = GetBfieldFullModelName('Planet', 'Jupiter', 'Model', 'JRM33', 'Type', 'core');
%   JRM09
%       Model = GetBfieldFullModelName('Planet', 'Jupiter', 'Model', 'JRM09', 'Type', 'core');
%   Cassini11
%       Model = GetBfieldFullModelName('Planet', 'Saturn', 'Model', 'Cassini11', 'Type', 'core');
%   Cassini11plus
%       Model = GetBfieldFullModelName('Planet', 'Saturn', 'Model', 'Cassini11plus', 'Type', 'core');
%
    arguments
        MNparam.Planet          char    {mustBeMember(MNparam.Planet, ...
                                            {'Earth', 'Jupiter', 'Saturn'})}
        MNparam.Model           char    {mustBeMember(MNparam.Model, ...
                                            {'IGRF', 'CHAOS', 'CM', 'COV-OBS', 'LCS', 'DIFI', 'SIFM', 'JRM33', 'JRM09', 'Cassini11', 'Cassini11plus'})}
        MNparam.Ver                     = []
        MNparam.Type            char    {mustBeMember(MNparam.Type, {'core', 'static', 'ionosphere'})}
        MNparam.Target          char    {mustBeMember(MNparam.Target, ...
                                            {'', 'Location', 'Name', 'TxtFile', 'MatFile', 'TxtFileLoc', 'MatFileLoc'})}    = ''
    end

    % Cross-check between model name and model version
    modelVSversion = containers.Map( ...
        {'IGRF', 'CHAOS', 'CM', 'COV-OBS', 'LCS', 'DIFI', 'SIFM', 'JRM33', 'JRM09', 'Cassini11', 'Cassini11plus'}, ...
        {[13 14], [7.18 8.5],    6,         2,     1,      6,     [],     [],     [],     [],     []} ...
    );
    version = modelVSversion(MNparam.Model);
    if ~isempty(version)
        if isempty(MNparam.Ver) || ~ismember(MNparam.Ver, version)
            error(['GetBfieldFullModelName: wrong version of the ' MNparam.Model ' model.' ...
            'Possible ' num2str(version)])
        end
    end
    % Get Magnetospheric models location
    Model.Location = fileparts(fileparts(which('GetBfieldFullModelName')));

    % Create structure >Model< with model name, txt & mat-files names
    if strcmp(MNparam.Model, 'IGRF')
        if ~strcmp(MNparam.Planet, 'Earth')
            error('GetBfieldFullModelName: set Planet = Earth for IGRF');
        end
        Model.Name = [MNparam.Model num2str(MNparam.Ver)];
        if strcmp(MNparam.Type, 'core')
            Model.TxtFile = [lower(Model.Name) 'coeffs.txt'];
            Model.MatFile = [lower(Model.Name) 'coeffs.mat'];
        else
            error('GetBfieldFullModelName: set Type = core for IGRF')
        end
    end

    if strcmp(MNparam.Model, 'CHAOS')
        if ~strcmp(MNparam.Planet, 'Earth')
            error('GetBfieldFullModelName: set Planet = Earth for CHAOS');
        end
        Model.Name = [MNparam.Model '-' num2str(MNparam.Ver)];
        if strcmp(MNparam.Type, 'ionosphere')
            error('GetBfieldFullModelName: set Type = core/static for CHAOS')
        else
            Model.TxtFile = [Model.Name '_' MNparam.Type '.shc.txt'];
            Model.MatFile = [Model.Name '_' MNparam.Type '.mat'];
        end
    end

   if strcmp(MNparam.Model, 'CM')
        if ~strcmp(MNparam.Planet, 'Earth')
            error('GetBfieldFullModelName: set Planet = Earth for CM');
        end
        Model.Name = [MNparam.Model num2str(MNparam.Ver)];
        if strcmp(MNparam.Type, 'core')
            Model.TxtFile = ['MCO_' Model.Name '.shc.txt'];
            Model.MatFile = ['MCO_' Model.Name '.mat'];
        elseif strcmp(MNparam.Type, 'static')
            Model.TxtFile = ['MLI_' Model.Name '.shc.txt'];
            Model.MatFile = ['MLI_' Model.Name '.mat'];
        elseif strcmp(MNparam.Type, 'ionosphere')
            Model.TxtFile = ['MIO_' Model.Name '.DBL.txt'];
            Model.MatFile = ['MIO_' Model.Name '.mat'];
        end
    end

    if strcmp(MNparam.Model, 'COV-OBS')
        if ~strcmp(MNparam.Planet, 'Earth')
            error('GetBfieldFullModelName: set Planet = Earth for COV-OBS');
        end
        Model.Name = [MNparam.Model '.x' num2str(MNparam.Ver) '-int'];
        Model.TxtFile = [Model.Name '.shc.txt'];
        Model.MatFile = [Model.Name '.mat'];
    end

    if strcmp(MNparam.Model, 'LCS')
        if ~strcmp(MNparam.Planet, 'Earth')
            error('GetBfieldFullModelName: set Planet = Earth for LCS');
        end
        Model.Name = [MNparam.Model '-' num2str(MNparam.Ver)];
        if strcmp(MNparam.Type, 'static')
            Model.TxtFile = [Model.Name '.shc.txt'];
            Model.MatFile = [Model.Name '.mat'];
        else
            error('GetBfieldFullModelName: set Type = static for LCS')
        end
    end

    if strcmp(MNparam.Model, 'DIFI')
        if ~strcmp(MNparam.Planet, 'Earth')
            error('GetBfieldFullModelName: set Planet = Earth for DIFI');
        end
        Model.Name = [MNparam.Model num2str(MNparam.Ver)];
        if strcmp(MNparam.Type, 'ionosphere')
            Model.TxtFile = [Model.Name '.txt'];
            Model.MatFile = [Model.Name '.mat'];
        else
            error('GetBfieldFullModelName: set Type = ionosphere for DIFI')
        end
    end

    if strcmp(MNparam.Model, 'SIFM')
        if ~strcmp(MNparam.Planet, 'Earth')
            error('GetBfieldFullModelName: set Planet = Earth for SIFM');
        end
        Model.Name = MNparam.Model;
        if strcmp(MNparam.Type, 'ionosphere')
            error('GetBfieldFullModelName: set Type = core/static for SIFM')
        else
            Model.TxtFile = [Model.Name '.shc.txt'];
            Model.MatFile = [Model.Name '_' MNparam.Type '.mat'];
        end
    end

    if strcmp(MNparam.Model, 'JRM33')
        if ~strcmp(MNparam.Planet, 'Jupiter')
            error('GetBfieldFullModelName: set Planet = Jupiter for JRM33');
        end
        Model.Name = MNparam.Model;
        if strcmp(MNparam.Type, 'core')
            Model.TxtFile = [Model.Name '_' MNparam.Type '.shc.txt'];
            Model.MatFile = [Model.Name '_' MNparam.Type '.mat'];
        else
            error('GetBfieldFullModelName: set Type = core for JRM33')
        end
    end

    if strcmp(MNparam.Model, 'JRM09')
    if ~strcmp(MNparam.Planet, 'Jupiter')
            error('GetBfieldFullModelName: set Planet = Jupiter for JRM09');
        end
        Model.Name = MNparam.Model;
        if strcmp(MNparam.Type, 'core')
            Model.TxtFile = [Model.Name '_' MNparam.Type '.shc.txt'];
            Model.MatFile = [Model.Name '_' MNparam.Type '.mat'];
        else
            error('GetBfieldFullModelName: set Type = core for JRM09')
        end
    end

    if strcmp(MNparam.Model, 'Cassini11')
    if ~strcmp(MNparam.Planet, 'Saturn')
            error('GetBfieldFullModelName: set Planet = Saturn for Cassini11');
        end
        Model.Name = MNparam.Model;
        if strcmp(MNparam.Type, 'core')
            Model.TxtFile = [Model.Name '_' MNparam.Type '.shc.txt'];
            Model.MatFile = [Model.Name '_' MNparam.Type '.mat'];
        else
            error('GetBfieldFullModelName: set Type = core for Cassini11')
        end
    end

    if strcmp(MNparam.Model, 'Cassini11plus')
    if ~strcmp(MNparam.Planet, 'Saturn')
            error('GetBfieldFullModelName: set Planet = Saturn for Cassini11plus');
        end
        Model.Name = MNparam.Model;
        if strcmp(MNparam.Type, 'core')
            Model.TxtFile = [Model.Name '_' MNparam.Type '.shc.txt'];
            Model.MatFile = [Model.Name '_' MNparam.Type '.mat'];
        else
            error('GetBfieldFullModelName: set Type = core for Cassini11plus')
        end
    end

    % Set location of txt & mat-files
    Model.TxtFileLoc = [Model.Location '/' MNparam.Planet '/' Model.Name '/' Model.TxtFile];
    Model.MatFileLoc = [Model.Location '/' MNparam.Planet '/' Model.Name '/' Model.MatFile];

    if ~exist(Model.TxtFileLoc, 'file') 
        error('GetBfieldFullModelName: Txt file with model not found')
    end
    %if ~exist(Model.MatFileLoc, 'file')
    %    error('GetBfieldFullModelName: Mat file with model not found')
    %end

    if ~isempty(MNparam.Target)
        Model = Model.(MNparam.Target);
    end