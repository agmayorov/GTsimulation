function PlotGTtrack(Track)
    arguments
        Track.GTtrack
        Track.MarkerSize          (1,1)   double  {mustBePositive(Track.MarkerSize)}                    = 8
        Track.MarkerEdgeColor                                                                           = 'r'
        Track.MarkerFaceColor                                                                           = [0.75,0.75,0.75]
        Track.LineColor                                                                                 = 'Blue1'
        Track.LineWidth           (1,1)   double  {mustBePositive(Track.LineWidth)}                     = 2
        Track.Event               (:,1)   double  {mustBePositive(Track.Event)}                         = 1
        Track.FieldLine                   char    {mustBeMember(Track.FieldLine, {'', 'on', 'off'})}   	= 'off'
    end

    PS = PLOT_STANDARDS();
    if ischar(Track.MarkerEdgeColor)
        if length(Track.MarkerEdgeColor) > 1
            Track.MarkerEdgeColor = PS.(Track.MarkerEdgeColor);
        end
    end
    if ischar(Track.MarkerFaceColor)
        if length(Track.MarkerFaceColor) > 1
            Track.MarkerFaceColor = PS.(Track.MarkerFaceColor);
        end
    end
    if ischar(Track.LineColor)
        if length(Track.LineColor) > 1
            Track.LineColor = PS.(Track.LineColor);
        end
    end
    
    for iev = 1:Track.Event
        X = Track.GTtrack(iev).data.Track.X;
        Y = Track.GTtrack(iev).data.Track.Y;
        Z = Track.GTtrack(iev).data.Track.Z;
 
        pfig = plot3(X, Y, Z, 'b', X(1), Y(1), Z(1), 'o', ...
            'MarkerSize', Track.MarkerSize, ...
            'MarkerEdgeColor', Track.MarkerEdgeColor, ...
            'MarkerFaceColor', Track.MarkerFaceColor);
        set(pfig, 'Color', Track.LineColor, 'LineWidth', Track.LineWidth);
        hold on
    end

    axis equal, box on, grid on, hold on

    if strcmp(Track.FieldLine, 'on')
        Rfl = Track.GTtrack(iev).data.GTmag.GuidingCentre.Rline;
        plot3(Rfl(:,1), Rfl(:,2), Rfl(:,3), '.b')
    end
    