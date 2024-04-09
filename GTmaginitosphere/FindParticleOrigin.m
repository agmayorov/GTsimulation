function ParticleOrigin = FindParticleOrigin(fGTtrack)
    %% Forward trajectory
    fGTparam = fGTtrack.data.GTparam;
    %PlotGTtrack('GTtrack', fGTtrack, 'LineColor', 'Red1')
    [f, lon_f] = BCpar(fGTtrack.data.BC); 
    [~, lon_total, ~, Nm, I1, ~, I2, ~] = ...
                    AddTrajectory(f, 0, 0, lon_f, fGTtrack.data.GTmag, 0, [], [], 1);
    [Rf, Vf] = GetLastPoints(fGTtrack, 1);
    clear fGTtrack
    
    %% Backward trajectory    
    bGTparam = RefreshGTparam('GTparam', fGTparam, ...
                'Vo', {'Backtracing', 'on'}, ...
                'IOinfo', {'Nfiles', 1, 'SaveDir', '', 'SaveFile', '', 'LoadGTtrack', 'off'}, ...
                'GTmag', {'TrackParam', 'on', 'ParticleOrigin', 'off', 'IsFirstRun', 0}, ...
                'BC', {'MaxRev', 5}, 'Nevents', 1, 'Verbose', 0);
    bGTtrack = RunGetTrajectoryInEMField(bGTparam, SetGTparticle(bGTparam));
    %PlotGTtrack('GTtrack', bGTtrack, 'LineColor', 'Blue1')
    [b, lon_b] = BCpar(bGTtrack.data.BC);
    [InitEndFlag, lon_total, is_full_revolution, Nm, I1, I1_disp, I2, I2_disp] = ...
                    AddTrajectory(f, b, lon_total, lon_b, bGTtrack.data.GTmag, Nm, I1, I2, -1);
    [Rb, Vb] = GetLastPoints(bGTtrack, -1);
    clear bGTtrack bGTparticle
        
    %% Determine the origin of the particle
    particle_origin = GetParticleOrigin(InitEndFlag, is_full_revolution, Nm, I1_disp, I2_disp);

    %% Repeat procedure
    while particle_origin == 6
        % Trace extension
        if f == 3
            s = 1;
            fGTparam = RefreshGTparam('GTparam', fGTparam, ...
                'Ro', {'LLA', [0 0 0], 'Center', Rf, 'CRFinp', 'geo', 'Units', 'm'}, 'Vo', {'Direction', Vf}, ...
                'GTmag', {'TrackParam', 'on', 'ParticleOrigin', 'off', 'IsFirstRun', 0}, ...
                'BC', {'MaxRev', 5}, 'Verbose', 0);
            fGTtrack = RunGetTrajectoryInEMField(fGTparam, SetGTparticle(fGTparam));
            [Rf, Vf] = GetLastPoints(fGTtrack, 1);
            %PlotGTtrack('GTtrack', fGTtrack, 'LineColor', 'Red1')
            [f, lon] = BCpar(fGTtrack.data.BC);
            GTtrack = fGTtrack;
            clear fGTtrack
        else
            if b == 3
                s = -1;
                bGTparam = RefreshGTparam('GTparam', bGTparam, ...
                    'Ro', {'LLA', [0 0 0], 'Center', Rb, 'CRFinp', 'geo', 'Units', 'm'}, 'Vo', {'Direction', Vb});
                bGTtrack = RunGetTrajectoryInEMField(bGTparam, SetGTparticle(bGTparam));
                [Rb, Vb] = GetLastPoints(bGTtrack, -1);
                %PlotGTtrack('GTtrack', bGTtrack, 'LineColor', 'Blue1')
                [b, lon] = BCpar(bGTtrack.data.BC);
                GTtrack = bGTtrack;
                clear bGTtrack
            else
                break
            end
        end
   
        [InitEndFlag, lon_total, is_full_revolution, Nm, I1, I1_disp, I2, I2_disp] = ...
                    AddTrajectory(f, b, lon_total, lon, GTtrack.data.GTmag, Nm, I1, I2, s);
                
        particle_origin = GetParticleOrigin(InitEndFlag, is_full_revolution, Nm, I1_disp, I2_disp);
    end
    
    Ind2Name = containers.Map( ...
        {         1,         2,               3,              4,         5,          6}, ...
        {'Galactic',  'Albedo', 'Quasi-trapped', 'Presipitated', 'Trapped', 'TraceExt'} ...
    );

    ParticleOrigin.Ind = particle_origin;
    ParticleOrigin.Name = Ind2Name(ParticleOrigin.Ind);
end

%%
function [R, V] = GetLastPoints(GTtrack, s)
    ind = [length(GTtrack.data.Track.X)-1 length(GTtrack.data.Track.X)];
    
    X =  GTtrack.data.Track.X(ind);
    Y =  GTtrack.data.Track.Y(ind);
    Z =  GTtrack.data.Track.Z(ind);
    
    R = [X(end) Y(end) Z(end)];    
    V = [X(end) - X(end-1), Y(end) - Y(end-1), Z(end) - Z(end-1)]./norm([X(end) - X(end-1), Y(end) - Y(end-1), Z(end) - Z(end-1)]);
    
    if s == -1
        V = -V;
    end
end

%% Add backward trajectory to forward one
function [InitEndFlag, lon_total, is_full_revolution, Nm, I1, I1_disp, I2, I2_disp] = ...
                    AddTrajectory(f, b, lon_total, lon, GTmag, Nm, I1, I2, s)
                
    InitEndFlag = {f, b};
    lon_total = lon_total + lon;
    is_full_revolution = 0 + (lon_total > 2*pi);
    
    % I1
    if s == 1
        I1 = [I1' GTmag.Invariants.I1']';
    else
        I1 = [fliplr(GTmag.Invariants.I1(1:end-1)') I1']';
    end
    d = (sum(I1) / length(I1) - I1).^2;
    I1_disp = sqrt(sum(d)/length(d)) / (sum(I1)/length(I1));
    clear first_invariant d
    
    % Mirror points
    Nm = [Nm GTmag.MirrorPoints.NumBo];
    
    % I2
    if s == 1
        I2 = [I2' GTmag.Invariants.I2']';
    else
        I2 = [fliplr(GTmag.Invariants.I2(1:end-1)') I2']';
    end
    d = (sum(I2) / length(I2) - I2).^2;
    I2_disp = sqrt(sum(d)/length(d)) / (sum(I2)/length(I2));
    clear second_invariant d
end  
%%
function [u, lon_u] = BCpar(BC)
    u = 1 * strcmp(BC.Status, 'DefaultBC_Rmin') + ...
        2 * strcmp(BC.Status, 'DefaultBC_Rmax') + ...
        3 * (strcmp(BC.Status, 'UserBCFunction') || ...
        strcmp(BC.Status, 'done'));
    lon_u = BC.lon_total;
end

%%
function particle_origin = GetParticleOrigin(InitEndFlag, is_full_revolution, num_mirror, first_invariant_disp, second_invariant_disp)
    first_invariant_disp_tol = 0.3;
    first_invariant_disp_tol_2 = 0.7;
    second_invariant_disp_tol = 0.3;
    full_revolution_tol = 5;

    if InitEndFlag{1} == 2
        particle_origin = 1;
        return
    end
    if InitEndFlag{1} == 1 && InitEndFlag{2} == 2
        particle_origin = 2;
        return
    end
    if InitEndFlag{1} == 3 && InitEndFlag{2} == 2
        particle_origin = 6;
        return
    end    
    if ~is_full_revolution && (InitEndFlag{1} == 3 && InitEndFlag{2} == 3)
        particle_origin = 6;
        return
    end
    
    if ~is_full_revolution
        if InitEndFlag{1} == 1 && InitEndFlag{2} == 1
            if isempty(num_mirror)
                particle_origin = 2;
            end
            if length(num_mirror) >= 1 && length(num_mirror) <= 2
                if first_invariant_disp < first_invariant_disp_tol_2
                    particle_origin = 4;
                else
                    particle_origin = 2;
                end
            end
            if length(num_mirror) > 2
                if first_invariant_disp < first_invariant_disp_tol_2
                    particle_origin = 3;
                else
                    particle_origin = 2;
                end
            end
            return
        end
        if length(num_mirror) <= 2
            particle_origin = 6;
        else
            if first_invariant_disp < first_invariant_disp_tol_2
                particle_origin = 3;
            else
                particle_origin = 2;
            end
        end
        return    
    end
    if is_full_revolution
        if InitEndFlag{1} == 1 || InitEndFlag{2} == 1
            if first_invariant_disp < first_invariant_disp_tol_2
                particle_origin = 3;
            else
                particle_origin = 2;
            end
            return
        end
        if (first_invariant_disp < first_invariant_disp_tol) && (second_invariant_disp < second_invariant_disp_tol) || (is_full_revolution >= full_revolution_tol)
            particle_origin = 5;
        else
            particle_origin = 6;
        end
    end
end