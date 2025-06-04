function Model = GetBfieldFullModelName(MNparam)
%   Function to get full name of Earth MF Model
%   Ver. 1, red. 3 / 28 June 2023 / A. Mayorov
%
%   Possible cases:
%   IGRF
%       Model = GetBfieldFullModelName('Model', 'IGRF', 'Ver', 13, 'Type', 'core');
%       Model = GetBfieldFullModelName('Model', 'IGRF', 'Ver', 13, 'Type', 'core', 'Target', 'TxtFileLoc');
%   CHAOS
%       Model = GetBfieldFullModelName('Model', 'CHAOS', 'Ver', 7.13, 'Type', 'core');
%       Model = GetBfieldFullModelName('Model', 'CHAOS', 'Ver', 7.13, 'Type', 'static');
%   CM
%       Model = GetBfieldFullModelName('Model', 'CM', 'Ver', 6, 'Type', 'core');
%       Model = GetBfieldFullModelName('Model', 'CM', 'Ver', 6, 'Type', 'static');
%       Model = GetBfieldFullModelName('Model', 'CM', 'Ver', 6, 'Type', 'ionosphere');
%   COV-OBS
%       Model = GetBfieldFullModelName('Model', 'COV-OBS', 'Ver', 2, 'Type', 'core');
%   LCS
%       Model = GetBfieldFullModelName('Model', 'LCS', 'Ver', 1, 'Type', 'static');
%   DIFI
%       Model = GetBfieldFullModelName('Model', 'DIFI', 'Ver', 6, 'Type', 'ionosphere');
%   SIFM
%       Model = GetBfieldFullModelName('Model', 'SIFM', 'Type', 'core');
%       Model = GetBfieldFullModelName('Model', 'SIFM', 'Type', 'static');
%
    arguments
        MNparam.Model           char    {mustBeMember(MNparam.Model, ...
                                            {'IGRF', 'CHAOS', 'CM', 'COV-OBS', 'LCS', 'DIFI', 'SIFM'})}
        MNparam.Ver                     = []
        MNparam.Type            char    {mustBeMember(MNparam.Type, {'core', 'static', 'ionosphere'})}
        MNparam.Target          char    {mustBeMember(MNparam.Target, ...
                                            {'', 'Location', 'Name', 'TxtFile', 'MatFile', 'TxtFileLoc', 'MatFileLoc'})}    = ''
    end
    
    % Cross-check between model name and model version
    modelVSverion = containers.Map( ...
        {'IGRF', 'CHAOS', 'CM', 'COV-OBS', 'LCS', 'DIFI', 'SIFM'}, ...
        {[13 14], [7.13 8.1],    6,         2,     1,      6,     []} ...
    );
    version = modelVSverion(MNparam.Model);
    if ~ismember(MNparam.Ver, version)
        error(['GetBfieldFullModelName: wrong version of the ' MNparam.Model ' model.' ...
            'Possible ' num2str(version)])
    end

    % Get Magnetospheric models location
    Model.Location = fileparts(which('GetBfieldFullModelName'));

    % Create structure >Model< with model name, txt & mat-files names
    if strcmp(MNparam.Model, 'IGRF')
        Model.Name = [MNparam.Model num2str(MNparam.Ver)];
        if strcmp(MNparam.Type, 'core')
            Model.TxtFile = [lower(Model.Name) 'coeffs.txt'];
            Model.MatFile = [lower(Model.Name) 'coeffs.mat'];
        else
            error('GetBfieldFullModelName: set Type = core for IGRF')
        end
    end
    
    if strcmp(MNparam.Model, 'CHAOS')
        Model.Name = [MNparam.Model '-' num2str(MNparam.Ver)];
        if strcmp(MNparam.Type, 'ionosphere')
            error('GetBfieldFullModelName: set Type = core/static for CHAOS')
        else
            Model.TxtFile = [Model.Name '_' MNparam.Type '.shc.txt'];
            Model.MatFile = [Model.Name '_' MNparam.Type '.mat'];
        end
    end
    
   if strcmp(MNparam.Model, 'CM')
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
        Model.Name = [MNparam.Model '.x' num2str(MNparam.Ver) '-int'];
        Model.TxtFile = [Model.Name '.shc.txt'];
        Model.MatFile = [Model.Name '.mat'];
    end

    if strcmp(MNparam.Model, 'LCS')
        Model.Name = [MNparam.Model '-' num2str(MNparam.Ver)];
        if strcmp(MNparam.Type, 'static')
            Model.TxtFile = [Model.Name '.shc.txt'];
            Model.MatFile = [Model.Name '.mat'];
        else
            error('GetBfieldFullModelName: set Type = static for LCS')
        end
    end
    
    if strcmp(MNparam.Model, 'DIFI')
        Model.Name = [MNparam.Model num2str(MNparam.Ver)];
        if strcmp(MNparam.Type, 'ionosphere')
            Model.TxtFile = [Model.Name '.txt'];
            Model.MatFile = [Model.Name '.mat'];
        else
            error('GetBfieldFullModelName: set Type = ionosphere for DIFI')
        end
    end
    
    if strcmp(MNparam.Model, 'SIFM')
        Model.Name = MNparam.Model;
        if strcmp(MNparam.Type, 'ionosphere')
            error('GetBfieldFullModelName: set Type = core/static for SIFM')
        else
            Model.TxtFile = [Model.Name '.shc.txt'];
            Model.MatFile = [Model.Name '_' MNparam.Type '.mat'];
        end
    end

    % Set location of txt & mat-files
    Model.TxtFileLoc = [Model.Location '/' Model.Name '/' Model.TxtFile];
    Model.MatFileLoc = [Model.Location '/' Model.Name '/' Model.MatFile];

    if ~exist(Model.TxtFileLoc, 'file') 
        error('GetBfieldFullModelName: Txt file with model not found')
    end
    %if ~exist(Model.MatFileLoc, 'file')
    %    error('GetBfieldFullModelName: Mat file with model not found')
    %end

    if ~isempty(MNparam.Target)
        Model = Model.(MNparam.Target);
    end