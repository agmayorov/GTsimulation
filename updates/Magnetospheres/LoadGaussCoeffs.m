function coefs_otp = LoadGaussCoeffs(GCparam)
%   Function to Load Gauss coeffficients from initial file of Earth MF Model
%   Ver. 1, red. 2 / 28 June 2023 / A. Mayorov
%
%   Examples
%       coefs = LoadGaussCoeffs('Model', 'IGRF', 'Ver', 13, 'Type', 'core', 'Date', [2006 06 15]);
%       coefs = LoadGaussCoeffs('Model', 'CHAOS', 'Ver', 7.13, 'Type', 'core', 'Date', [2006 06 15], 'gh', 1);
%
    arguments
        GCparam.Model           char    {mustBeMember(GCparam.Model, ...
                                            {'IGRF', 'CHAOS', 'CM', 'COV-OBS', 'LCS', 'SIFM', 'DIFI'})}
        GCparam.Ver     (1,1)   double  {mustBePositive(GCparam.Ver)}
        GCparam.Type            char    {mustBeMember(GCparam.Type, {'core', 'static', 'ionosphere'})}
        GCparam.Date            double  {mustBePositive(GCparam.Date)}          = 0
        GCparam.gh              double  {mustBeMember(GCparam.gh, [0 1])}       = 0
    end
    
    Model = GetBfieldFullModelName('Model', GCparam.Model, 'Ver', GCparam.Ver, 'Type', GCparam.Type);

    % Convert time to fractional years
    if ismember(length(GCparam.Date), [3 6])
        DTnum = datenum(GCparam.Date);
    elseif length(GCparam.Date) == 1
        DTnum = GCparam.Date;
    else
        error('LoadGaussCoeffs: Date is wrongly defined');
    end
        
    timevec = datevec(DTnum);
    time = timevec(1) + (DTnum - datenum([timevec(1) 1 1]))./(365 + double(...
        (~mod(timevec(1),4) & mod(timevec(1),100)) | (~mod(timevec(1),400))));

    % Load coefs and years variables
    load(Model.MatFileLoc)
    yrs = cell2mat({coefs.year});
    nyrs = numel(yrs);
    
    if nyrs == 1
        if GCparam.gh == 0
            coefs_otp.g = coefs.g;
            coefs_otp.h = coefs.h;
        else
            coefs_otp.gh = coefs.gh;
        end

    else
        % Check validity on time.    
        if time < yrs(1) || time > yrs(end)
            error('LoadGaussCoeffs:timeOutOfRange', [GCparam.Model ' is only valid between ' ...
                num2str(yrs(1)) ' and ' num2str(yrs(end))]);
        end

        % Get the nearest epoch that the current time is between.
        lastepoch = find(yrs - time < 0, 1, 'last');
        if isempty(lastepoch)
            lastepoch = 1;
        end
        nextepoch = lastepoch + 1;

        % Output either g and h matrices or gh vector
        if GCparam.gh == 0

            % Get the coefficients based on the epoch.
            lastg = coefs(lastepoch).g; lasth = coefs(lastepoch).h;
            nextg = coefs(nextepoch).g; nexth = coefs(nextepoch).h;

            % If one of the coefficient matrices is smaller than the other, enlarge
            % the smaller one with 0's.
            if size(lastg, 1) > size(nextg, 1)
                smalln = size(nextg, 1);
                nextg = zeros(size(lastg));
                nextg(1:smalln, (0:smalln)+1) = coefs(nextepoch).g;
                nexth = zeros(size(lasth));
                nexth(1:smalln, (0:smalln)+1) = coefs(nextepoch).h;
            elseif size(lastg, 1) < size(nextg, 1)
                smalln = size(lastg, 1);
                lastg = zeros(size(nextg));
                lastg(1:smalln, (0:smalln)+1) = coefs(lastepoch).g;
                lasth = zeros(size(nexth));
                lasth(1:smalln, (0:smalln)+1) = coefs(lastepoch).h;
            end

            % Calculate g and h using a linear interpolation between the last and
            % next epoch.
            if isfield(coefs(nextepoch), 'slope') && coefs(nextepoch).slope
                gslope = nextg;
                hslope = nexth;
            else
                gslope = (nextg - lastg)/diff(yrs([lastepoch nextepoch]));
                hslope = (nexth - lasth)/diff(yrs([lastepoch nextepoch]));
            end
            coefs_otp.g = lastg + gslope*(time - yrs(lastepoch));
            coefs_otp.h = lasth + hslope*(time - yrs(lastepoch));

        else

            % Get the coefficients based on the epoch.
            lastgh = coefs(lastepoch).gh;
            nextgh = coefs(nextepoch).gh;

            % If one of the coefficient vectors is smaller than the other, enlarge
            % the smaller one with 0's.
            if length(lastgh) > length(nextgh)
                smalln = length(nextgh);
                nextgh = zeros(size(lastgh));
                nextgh(1:smalln) = coefs(nextepoch).gh;
            elseif length(lastgh) < length(nextgh)
                smalln = length(lastgh);
                lastgh = zeros(size(nextgh));
                lastgh(1:smalln) = coefs(lastepoch).gh;
            end

            % Calculate gh using a linear interpolation between the last and next
            % epoch.
            if isfield(coefs(nextepoch), 'slope') && coefs(nextepoch).slope
                ghslope = nextgh;
            else
                ghslope = (nextgh - lastgh)/diff(yrs([lastepoch nextepoch]));
            end
            coefs_otp.gh = lastgh + ghslope*(time - yrs(lastepoch));

        end
    end
