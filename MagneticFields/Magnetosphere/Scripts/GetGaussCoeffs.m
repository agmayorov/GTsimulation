function GetGaussCoeffs(GCparam)
%   Function to get Gauss coefficients from initial file of Earth MF Model
%   Ver. 1, red. 1 / 17 April 2023 / A. Mayorov
%   Data teken from http://www.spacecenter.dk/files/magnetic-models/
%                   https://www.space.dtu.dk/english/research/scientific_data_and_models/magnetic_field_models
%                   https://geomag.colorado.edu/difi-6
%
%   Examples
%       GetGaussCoeffs('Model', 'IGRF', 'Ver', 13);
%       GetGaussCoeffs('Model', 'CHAOS', 'Ver', 7.13);
%       GetGaussCoeffs('Model', 'CM', 'Ver', 6);
%       GetGaussCoeffs('Model', 'COV-OBS', 'Ver', 2);
%       GetGaussCoeffs('Model', 'LCS', 'Ver', 1)
%       GetGaussCoeffs('Model', 'DIFI', 'Ver', 6)
%       GetGaussCoeffs('Model', 'SIFM')
%

%   TODO
%   - Ionospheric magnetic field for CM6, DIFI6
%   - Check qw in CM, COV-OBS, LCS, SIFM !!! qw - skipped g & h for
%   lithospheric MF

    arguments
        GCparam.Model           char    {mustBeMember(GCparam.Model, ...
                                            {'IGRF', 'CHAOS', 'CM', 'COV-OBS', 'LCS', 'DIFI','SIFM'})}
        GCparam.Ver     (1,1)   double  {mustBePositive(GCparam.Ver)}   = 1;
    end

    if strcmp(GCparam.Model, 'CHAOS') % Chenk number of lines in txt files
        % Core field model 1 to 20
        Model = GetBfieldFullModelName('Model', 'CHAOS', 'Ver', GCparam.Ver, 'Type', 'core');
        coefs = getcoefs(Model.TxtFileLoc, 230, 210, 0, 1, 20, 4, 1);
        save(Model.MatFile, 'coefs');
        
        % Lithospheric field model 21 to 185
        Model = GetBfieldFullModelName('Model', 'CHAOS', 'Ver', GCparam.Ver, 'Type', 'static');
        coefs = getcoefs(Model.TxtFileLoc, 17160, 16995, 440, 21, 185, 4, 1);
        save(Model.MatFile, 'coefs');
    end
    
    if strcmp(GCparam.Model, 'CM')
        % Core field model 1 to 18
        Model = GetBfieldFullModelName('Model', 'CM', 'Ver', GCparam.Ver, 'Type', 'core');
        coefs = getcoefs(Model.TxtFile, 189, 171, 99999999, 1, 18, 4, 1);
        save(Model.MatFile, 'coefs');
        
        % Lithospheric field model 14 to 120
        Model = GetBfieldFullModelName('Model', 'CM', 'Ver', GCparam.Ver, 'Type', 'static');
        coefs = getcoefs(Model.TxtFile, 7276, 7169, 99999999, 14, 120, 4, 1);
        save(Model.MatFile, 'coefs');
        
        % Ionosphere field model 14 to 60 / 1 to 12
%         Model = GetBfieldFullModelName('Model', 'CM', 'Ver', GCparam.Ver, 'Type', 'ionosphere');
%         coefs = getcoefs(Model.TxtFile, 1428, 1308, 99999999, 1, 60, 4, 1);
%         save(Model.MatFile, 'coefs');
    end
    
    if strcmp(GCparam.Model, 'COV-OBS')
        % Core field model 1 to 14
        Model = GetBfieldFullModelName('Model', 'COV-OBS', 'Ver', GCparam.Ver, 'Type', 'core');
        coefs = getcoefs(Model.TxtFile, 119, 105, 99999999, 1, 14, 7, 1);
        save(Model.MatFile, 'coefs');
    end

    if strcmp(GCparam.Model, 'LCS')
        % Lithospheric field model 1 to 185
        Model = GetBfieldFullModelName('Model', 'LCS', 'Ver', GCparam.Ver, 'Type', 'static');
        coefs = getcoefs(Model.TxtFile, 17389, 17205, 99999999, 1, 185, 4, 1);
        save(Model.MatFile, 'coefs');
    end
    
    if strcmp(GCparam.Model, 'SIFM')
        % Core field model 1 to 13
        Model = GetBfieldFullModelName('Model', 'SIFM', 'Ver', GCparam.Ver, 'Type', 'core');
        coefs = getcoefs(Model.TxtFile, 104, 91, 99999999, 1, 13, 6, 1);
        save(Model.MatFile, 'coefs');
        
        % Lithospheric field model 14 to 70
        Model = GetBfieldFullModelName('Model', 'SIFM', 'Ver', GCparam.Ver, 'Type', 'static');
        coefs = getcoefs(Model.TxtFile, 2457, 2388, 99999999, 14, 70, 6, 1);
        save(Model.MatFile, 'coefs');
    end
    
    if strcmp(GCparam.Model, 'IGRF') % Change initialization
        Model = GetBfieldFullModelName('Model', 'IGRF', 'Ver', GCparam.Ver, 'Type', 'core');
        fid = fopen(Model.TxtFileLoc);
        
        for l = 1:3
            fgetl(fid);
        end
        L = strsplit(fgetl(fid));
        year = cell2mat(cellfun(@str2num, L(4:end-1), 'uni', 0))';
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

        save(Model.MatFile, 'coefs');
    end
end

function coefs = getcoefs(filename, n, m, qw, kmin, kmax, r, p)
    fid = fopen(filename);
    
    for l = 1:r
        fgetl(fid);
    end
    if ~strcmp(filename, 'CM6/MIO_CM6.DBL.txt')
        L = strsplit(fgetl(fid));
        year = cell2mat(cellfun(@str2num, L, 'uni', 0))';
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