function GetGaussCoeffs(GCparam)
%   Function to get Gauss coefficients from initial file of a celestial body\ (planet or moon) MF Model
%   Ver. 1, red. 1 / 17 April 2023 / A. Mayorov
%   Data teken from http://www.spacecenter.dk/files/magnetic-models/
%                   https://www.space.dtu.dk/english/research/scientific_data_and_models/magnetic_field_models
%                   https://geomag.colorado.edu/difi-6
%
%   Examples
%       GetGaussCoeffs('Planet', 'Earth', 'Model', 'IGRF', 'Ver', 13);             #Planets
%       GetGaussCoeffs('Planet', 'Earth', 'Model', 'CHAOS', 'Ver', 7.13);
%       GetGaussCoeffs('Planet', 'Earth', 'Model', 'CM', 'Ver', 6);
%       GetGaussCoeffs('Planet', 'Earth', 'Model', 'COV-OBS', 'Ver', 2);
%       GetGaussCoeffs('Planet', 'Earth', 'Model', 'LCS', 'Ver', 1)
%       GetGaussCoeffs('Planet', 'Earth', 'Model', 'DIFI', 'Ver', 6)
%       GetGaussCoeffs('Planet', 'Earth', 'Model', 'SIFM')
%       GetGaussCoeffs('Planet', 'Jupiter', 'Model', 'JRM33')
%       GetGaussCoeffs('Planet', 'Jupiter', 'Model', 'JRM09')
%       GetGaussCoeffs('Planet', 'Saturn', 'Model', 'Cassini11')
%       GetGaussCoeffs('Planet', 'Saturn', 'Model', 'Cassini11plus')
%       GetGaussCoeffs('Planet', 'Uranus', 'Model', 'Q3')
%       GetGaussCoeffs('Planet', 'Neptune', 'Model', 'O8')
%       GetGaussCoeffs('Planet', 'Mercury', 'Model', 'MBF_a_n')
%       GetGaussCoeffs('Planet', 'Mars', 'Model', 'Langlais2019')
%       GetGaussCoeffs('Moon', 'Ganymede', 'Model', 'MagModel4')                #Moons
%

%   TODO
%   - Ionospheric magnetic field for CM6, DIFI6
%   - Check qw in CM, COV-OBS, LCS, SIFM !!! qw - skipped g & h for
%   lithospheric MF

    arguments
        GCparam.Planet          char    {mustBeMember(GCparam.Planet, ...
                                            {'', 'Earth', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Mercury', 'Mars'})}   = ''
        GCparam.Moon            char    {mustBeMember(GCparam.Moon, ...
                                            {'', 'Ganymede'})}   = ''
        GCparam.Model           char    {mustBeMember(GCparam.Model, ...
                                            {'IGRF', 'CHAOS', 'CM', 'COV-OBS', 'LCS', 'DIFI','SIFM', 'JRM33', 'JRM09', 'Cassini11', 'Cassini11plus', 'Q3', 'O8', 'MBF_a_n', 'Langlais2019', 'MagModel4'})}
        GCparam.Ver     (1,1)   double  {mustBePositive(GCparam.Ver)}   = 1;
    end

     % Determine celestial body from Planet or Moon input
    assert(~isempty(GCparam.Planet) + ~isempty(GCparam.Moon) == 1, ...
        'GetGaussCoeffs: specify exactly one of Planet or Moon')
    Body = GCparam.Planet;
    if isempty(Body), Body = GCparam.Moon; end

    % Determine keyword for GetBfieldFullModelName
    if ~isempty(GCparam.Planet)
        bodyKeyword = 'Planet';
    else
        bodyKeyword = 'Moon';
    end

    if strcmp(GCparam.Model, 'CHAOS') % Chenk number of lines in txt files
        % Core field model 1 to 20
        Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'CHAOS', 'Ver', GCparam.Ver, 'Type', 'core');
        coefs = getcoefs(Model.TxtFileLoc, 230, 210, 0, 1, 20, 4, 1);
        save(Model.MatFileLoc, 'coefs');

        % Lithospheric field model 21 to 185
        Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'CHAOS', 'Ver', GCparam.Ver, 'Type', 'static');
        coefs = getcoefs(Model.TxtFileLoc, 17160, 16995, 440, 21, 185, 4, 1);
        save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'CM')
        % Core field model 1 to 18
        Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'CM', 'Ver', GCparam.Ver, 'Type', 'core');
        coefs = getcoefs(Model.TxtFileLoc, 189, 171, 99999999, 1, 18, 4, 1);
        save(Model.MatFileLoc, 'coefs');

        % Lithospheric field model 14 to 120
        Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'CM', 'Ver', GCparam.Ver, 'Type', 'static');
        coefs = getcoefs(Model.TxtFileLoc, 7276, 7169, 99999999, 14, 120, 4, 1);
        save(Model.MatFileLoc, 'coefs');

        % Ionosphere field model 14 to 60 / 1 to 12
%         Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'CM', 'Ver', GCparam.Ver, 'Type', 'ionosphere');
%         coefs = getcoefs(Model.TxtFileLoc, 1428, 1308, 99999999, 1, 60, 4, 1);
%         save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'COV-OBS')
        % Core field model 1 to 14
        Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'COV-OBS', 'Ver', GCparam.Ver, 'Type', 'core');
        coefs = getcoefs(Model.TxtFileLoc, 119, 105, 99999999, 1, 14, 7, 1);
        save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'LCS')
        % Lithospheric field model 1 to 185
        Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'LCS', 'Ver', GCparam.Ver, 'Type', 'static');
        coefs = getcoefs(Model.TxtFileLoc, 17389, 17205, 99999999, 1, 185, 4, 1);
        save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'SIFM')
        % Core field model 1 to 13
        Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'SIFM', 'Ver', GCparam.Ver, 'Type', 'core');
        coefs = getcoefs(Model.TxtFileLoc, 104, 91, 99999999, 1, 13, 6, 1);
        save(Model.MatFileLoc, 'coefs');

        % Lithospheric field model 14 to 70
        Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'SIFM', 'Ver', GCparam.Ver, 'Type', 'static');
        coefs = getcoefs(Model.TxtFileLoc, 2457, 2388, 99999999, 14, 70, 6, 1);
        save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'IGRF') % Change initialization
        Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'IGRF', 'Ver', GCparam.Ver, 'Type', 'core');
        fid = fopen(Model.TxtFileLoc);

        for l = 1:3
            fgetl(fid);
        end
        L = strsplit(fgetl(fid));
        year = cell2mat(cellfun(@str2num, L(4:end-1), 'uni', 0));
        year(end+1) = year(end) + 5;

        [s, sg, sh, gN, gM, hN, hM, g, h, gh] = initialization(104, 91, 0, length(year));
        for n = 1:13
            for m = 0:n
                sg = sg+1;
                L = strsplit(fgetl(fid));
                gN(sg) = str2double(L{2});
                gM(sg) = str2double(L{3});
                g(sg, :) = cell2mat(cellfun(@str2num, L(4:end), 'uni', 0));
                s = s + 1;
                gh(s, :) = g(sg, :);
                if m ~= 0
                    sh = sh + 1;
                    L = strsplit(fgetl(fid));
                    hN(sh) = str2double(L{2});
                    hM(sh) = str2double(L{3});
                    h(sh, :) = cell2mat(cellfun(@str2num, L(4:end), 'uni', 0));
                    s = s + 1;
                    gh(s, :) = h(sh, :);
                end
            end
        end
        fclose(fid);

        coefs = setcoefs(year, g, gN, gM, h, hN, hM, gh, 13, 14);

        save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'JRM33')
       Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'JRM33', 'Type', 'core');
       coefs = getcoefs(Model.TxtFileLoc, 189, 171, 0, 1, 18, 2, 2);
       save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'JRM09')
       Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'JRM09', 'Type', 'core');
       coefs = getcoefs(Model.TxtFileLoc, 65, 55, 0, 1, 10, 2, 2);
       save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'Cassini11')
       Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'Cassini11', 'Type', 'core');
       coefs = getcoefs(Model.TxtFileLoc, 77, 66, 0, 1, 11, 2, 2);
       save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'Cassini11plus')
       Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'Cassini11plus', 'Type', 'core');
       coefs = getcoefs(Model.TxtFileLoc, 119, 105, 0, 1, 14, 2, 2);
       save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'Q3')
       Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'Q3', 'Type', 'core');
       coefs = getcoefs(Model.TxtFileLoc, 5, 3, 0, 1, 2, 2, 2);
       save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'O8')
       Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'O8', 'Type', 'core');
       coefs = getcoefs(Model.TxtFileLoc, 9, 6, 0, 1, 3, 2, 2);
       save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'MBF_a_n')
       Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'MBF_a_n', 'Type', 'core');
       coefs = getcoefs(Model.TxtFileLoc, 9, 6, 0, 1, 3, 3, 2);
       save(Model.MatFileLoc, 'coefs');
    end

     if strcmp(GCparam.Model, 'Langlais2019')
       Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'Langlais2019', 'Type', 'static');
       coefs = getcoefs(Model.TxtFileLoc, 9179, 9045, 0, 1, 134, 2, 1);
       save(Model.MatFileLoc, 'coefs');
    end

    if strcmp(GCparam.Model, 'MagModel4')
       Model = GetBfieldFullModelName(bodyKeyword, Body, 'Model', 'MagModel4', 'Type', 'core');
       coefs = getcoefs(Model.TxtFileLoc, 5, 3, 0, 1, 2, 2, 1);
       save(Model.MatFileLoc, 'coefs');
    end
end

function coefs = getcoefs(filename, n, m, qw, kmin, kmax, r, p)
    fid = fopen(filename);

    for l = 1:r
        fgetl(fid);
    end
    if ~strcmp(filename, 'CM6/MIO_CM6.DBL.txt')
        L = strsplit(fgetl(fid));
        year = cell2mat(cellfun(@str2num, L, 'uni', 0));
    else
        year = 1999.0:0.5:2023.5;
    end

    %[s, sg, sh] = deal(0);
    [s, sg, sh, gN, gM, hN, hM, g, h, gh] = initialization(n, m, qw, length(year));
    for n = 1:kmax
        for m = -n:n
            s = s + 1;
            if n < kmin
                continue
            end
            if feof(fid)
            	break
            end            
            
            L = strsplit(fgetl(fid));
            if isempty(L{end})
                L(end) = [];
            end
            if isempty(L{1})
                L(1) = [];
            end
            
            if str2double(L{p+1}) >= 0
                sg = sg+1;
                gN(sg) = str2double(L{p});
                gM(sg) = str2double(L{p+1});
                g(sg, :) = cell2mat(cellfun(@str2num, L(p+2:end), 'uni', 0));
                gh(s, :) = g(sg, :);
            else
                sh = sh + 1;
                hN(sh) = str2double(L{p});
                hM(sh) = abs(str2double(L{p+1}));
                h(sh, :) = cell2mat(cellfun(@str2num, L(p+2:end), 'uni', 0));
                gh(s, :) = h(sh, :);
            end
        end
    end
    fclose(fid);
    %[sg sh]
    
    coefs = setcoefs(year, g, gN, gM, h, hN, hM, gh, kmax, kmax+1);
end

function [s, sg, sh, gN, gM, hN, hM, g, h, gh] = initialization(n, m, qw, y)
    s = 0;
    [sg, sh] = deal(0);
    [gN, gM] = deal(n);
    [hN, hM] = deal(m);
    g  = zeros(n, y);
    h  = zeros(m, y);
    gh = zeros(n+m+qw, y); % zeros(n*(m+1), y); % zeros(n+m, y)
end

function coefs = setcoefs(year, g, gN, gM, h, hN, hM, gh, n, m)
    coefs = struct('year', [], 'g', [], 'h', [], 'gh', []);

    for idx = 1 : length(year)
        coefs(idx).year = year(idx);
        
        gmat = zeros(n, m);
        gmat(sub2ind([n, m], gN, gM+1)) = g(:, idx);
        coefs(idx).g = gmat;

        hmat = zeros(n, m);
        hmat(sub2ind([n, m], hN, hM+1)) = h(:, idx);
        coefs(idx).h = hmat;

        coefs(idx).gh = gh(:, idx);
    end
end