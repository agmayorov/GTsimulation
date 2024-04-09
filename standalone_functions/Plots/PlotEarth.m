function PlotEarth(varargin)
%   Ver. 1, red. 1 / 31 March 2023 / A. Mayorov

    % Load input argument
    PEProps = GetPEArgsStruct(varargin{:});
    [Dates.Year, Dates.DoY, Dates.Secs, Dates.DTnum] = YDS(PEProps.Date);

    % Load constants
    Units = mUnits;
    Re = Units.RE2km;
    Rm = Units.RM2km;

    % Number of globe panels around the equator 
    % deg/panel = 360/npanels / 180 by default
    npanels = 90;

    % Get Earth surface
    [x, y, z] = ellipsoid(0, 0, 0, Re, Re, Re, npanels);

    % Get atmosphere surface
    plotAtmo = 0;
    if strcmp(PEProps.Atmo, 'on')
        [xa, ya, za] = ellipsoid(0, 0, 0, ...
                                 Re + PEProps.AtmoHeight, Re + PEProps.AtmoHeight, Re + PEProps.AtmoHeight, ...
                                 npanels);
        plotAtmo = 1;
    end

    % Get Moon surface
    plotMoon = 0;
    if strcmp(PEProps.Moon, 'on')
        plotMoon = 1;
        [xm, ym, zm] = ellipsoid(PEProps.MoonPosition(1), PEProps.MoonPosition(2), PEProps.MoonPosition(3), ...
                                 Rm, Rm, Rm, npanels);
    end
    
    % Change CRF from GEO to input
    if ~isempty(PEProps.CRF)
        switch lower(PEProps.CRF)
            case 'mag'
                [x, y, z] = convert_mag(x, y, z, Dates, npanels);
                if plotAtmo
                    [xa, ya, za] = convert_mag(xa, ya, za, Dates, npanels);
                end
                if plotMoon
                    [xm, ym, zm] = convert_mag(xm, ym, zm, Dates, npanels);
                end
            case 'gsm'
                [x, y, z] = convert_gsm(x, y, z, Dates, npanels);
                if plotAtmo
                    [xa, ya, za] = convert_gsm(xa, ya, za, Dates, npanels);
                end
                if plotMoon
                    [xm, ym, zm] = convert_gsm(xm, ym, zm, Dates, npanels);
                end
        end
    end

    UserUnits = containers.Map( ...
        { 'm',       'km',       'RE'}, ...
        {1e-3,          1, Units.RE2km} ...
    );
    r = UserUnits(PEProps.Scale);

    % Plot Earth
    globe = surf(x/r, y/r, -z/r, 'FaceColor', 'none', 'EdgeColor', 0.5*[1 1 1]);
    set(globe, 'FaceColor', 'texturemap', 'CData', imread('Earth.jpg'), 'FaceAlpha', 1, 'EdgeColor', 'none');
    axis equal; hold on

    % Plot atmosphere
    if plotAtmo
        globea = surf(xa/r, ya/r, -za/r, 'FaceColor', 'none', 'EdgeColor', 0.5*[1 1 1]);
        set(globea, 'FaceColor', 'c', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    end

    % Plot Moon
    if plotMoon
        globe = surf(xm/r, ym/r, -zm/r, 'FaceColor', 'none', 'EdgeColor', 0.5*[1 1 1]);
        set(globe, 'FaceColor', 'texturemap', 'CData', imread('Moon.jpg'), 'FaceAlpha', 1, 'EdgeColor', 'none');
    end

    % Set axis etc.
    if strcmp(PEProps.Scale, 'RE')
        PEProps.Scale = 'R_{E}';
    end
    xlabel(['X, ' PEProps.Scale])
    ylabel(['Y, ' PEProps.Scale])
    zlabel(['Z, ' PEProps.Scale])
end

function [PEArgs] = GetPEArgsStruct(PEArgs)
    arguments
        PEArgs.CRF	                    char   = 'geo'
        PEArgs.Date             (1,3)   double = [2006 06 15]
		PEArgs.Atmo                     char   = 'off'
        PEArgs.AtmoHeight       (1,1)   double = 500
		PEArgs.Moon	                    char   = 'off'
        PEArgs.MoonPosition     (1,3)   double = [384400 0 0] % km in geo
        PEArgs.Scale                    char   = 'RE' % RE, km, m
    end
end

function [x, y, z] = mat2row(x, y, z)
    x = reshape(x, [numel(x), 1])';
    y = reshape(y, [numel(y), 1])';
    z = reshape(z, [numel(z), 1])';
end

function [x, y, z] = row2mat(x, y, z, npanels)
    x = reshape(x, [npanels+1, npanels+1]);
    y = reshape(y, [npanels+1, npanels+1]);
    z = reshape(z, [npanels+1, npanels+1]);
end

function [x, y, z] = convert_mag(x, y, z, Dates, npanels)
    [x, y, z] = mat2row(x, y, z);
    [x, y, z] = geo2mag_eccentric(x*1e3, y*1e3, z*1e3, 1, Dates.DTnum);
    [x, y, z] = row2mat(x/1e3, y/1e3, z/1e3, npanels);
end

function [x, y, z] = convert_gsm(x, y, z, Dates, npanels)
    [x, y, z] = mat2row(x, y, z);
    [x, y, z] = geo2gsm(x, y, z, Dates.Year, Dates.DoY, Dates.Secs, 1);
    [x, y, z] = row2mat(x, y, z, npanels);
end
