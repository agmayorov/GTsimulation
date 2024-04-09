function GTmag = GetGTtrackParam(R, H, GTparticle, GTparam)
    % Function to calculate:
    %   - first [kg*m^2*sec^-2*T^-1] and second adiabatic invariants
    %   - pitch-angle
    %   - mirror points position
    % 	- L-shell
    %   - guiding centre
    arguments
        R   (:,3)   double      % Coordinates, GEO, m
        H   (:,3)   double      % Magnetic field, nT
        GTparticle  struct      % GTparticle
        GTparam     struct      % GTparam
    end

    %% Units & Constants
    Units = mUnits;
    Const = mConstants;
    
    M = GTparticle.M*Units.MeV2kg;  % MeV/c2 to kg
    dt = GTparam.dt;            % Timestep, s
    EarthBfield = SetEarthBfield(GTparam.EMFF.Bfield, GTparam.Dates.Date);

    %% Healpful defenitions
	V = (R(2:end, :) - R(1:end-1, :)) / dt;
    Vn = vecnorm(V, 2, 2);
    Hn = vecnorm(H, 2, 2);
    VndotHn = Vn .* Hn;
    VdotH = V(:, 1) .* H(:, 1) + V(:, 2) .* H(:, 2) + V(:, 3) .* H(:, 3);
    Y = 1 ./ sqrt((1 - Vn.^2 / Const.c^2));
    
    %% GTtrack parameters
    % First invariant
    I1 = M * Y.^2 .* (Vn.^2 - (VdotH ./ Hn) .^ 2) ./ (2 * Hn);
    
    % Pitch-angle
    pitch = acos(VdotH./VndotHn)/pi*180;
    
    % Mirror points
    [num_mirror, num_eq_pitch, Hm, Heq, num_B0] = GetMirrorPoint(pitch, Hn);
    MirrorPoints.Num     = num_mirror;
    MirrorPoints.NumEq   = num_eq_pitch;
    MirrorPoints.Hm      = Hm;
    MirrorPoints.Heq     = Heq;
    MirrorPoints.NumBo   = num_B0;
    
    PitchAngles.pitch    = pitch;
    PitchAngles.pitch_eq = [];
    if num_eq_pitch
        PitchAngles.pitch_eq = pitch(num_eq_pitch);
    end
    
    % Second invariant
    I2 = GetSecondInvariant(R(:,1)', R(:,2)', R(:,3)', num_mirror, H(:,1)', H(:,2)', H(:,3)', Hn', Hm')';
    
    Invariants.I1 = I1;
    Invariants.I2 = I2;

    if GTparam.GTmag.IsFirstRun
        % L-shell
        Lshell = GetLshell(I2, Hm);
%        Lgen = sqrt(R(num_eq_pitch - 1, 1).^2 + R(num_eq_pitch - 1, 2).^2 + R(num_eq_pitch - 1, 3).^2)/Units.RE2m;
    
        % Magnetic field line of Guiding Centre
        LR = GetLarmorRadius(GTparticle.T, norm([H(1,1) H(1,2) H(1,3)]), GTparticle.Z, GTparticle.M, pitch(1));
        Nit = 2*pi*LR / (Vn(1)*dt);
        GuidingCentre.LR = LR;
        GuidingCentre.LRNit = Nit;
    
        Nit = min(Nit+1, length(R(:,1)));
        Nit = floor(1:Nit/3-1:Nit+1);
        Rmin = zeros(numel(Nit), 3);
        for i = 1:length(Nit)
            [Rline, Bline] = GetEarthBfieldline(R(Nit(i),:), EarthBfield, Units);
            Bn = vecnorm(Bline, 2, 2);
            [~, e] = min(Bn);
            Rmin(i, :) = Rline(e, :);
            if i == 1
                parReq = Rline(e,:);
                parBeq = Bline(e,:);
                parBBo = norm([H(1,1) H(1,2) H(1,3)])/norm(parBeq);
            end
            %Rline = Rline/Units.RE2m;
            %plot3(Rline(:, 1), Rline(:, 2), Rline(:, 3)); hold on
            clear Rline Bline Bn e
        end
        [Rline, Bline] = GetEarthBfieldline([mean(Rmin(:,1)) mean(Rmin(:,2)) mean(Rmin(:,3))], EarthBfield, Units);
        Bn = vecnorm(Bline, 2, 2);
        [~, e] = min(Bn);
        Req = Rline(e,:);
        Beq = Bline(e,:);
        BBo = norm([H(1,1) H(1,2) H(1,3)])/norm(Beq);
        %Rline = Rline/Units.RE2m; Req = Req/Units.RE2m;
        %plot3(Rline(:, 1), Rline(:, 2), Rline(:, 3)); hold on; scatter3(Req(1), Req(2), Req(3))
    
        % Field line of particle
        GuidingCentre.parReq = parReq;
        GuidingCentre.parBeq = parBeq;
        GuidingCentre.parBBo = parBBo;

        %R(1,:)
        %parReq

        [parReq(1), parReq(2), parReq(3)] = geo2mag_eccentric(parReq(1), parReq(2), parReq(3), 1, GTparam.Dates.DTnum);
        GuidingCentre.parL = norm(parReq)/Units.RE2m;

        % Field line of guiding centre
        GuidingCentre.Req = Req;
        GuidingCentre.Beq = Beq;
        GuidingCentre.BBo = BBo;
        [Req(1), Req(2), Req(3)] = geo2mag_eccentric(Req(1), Req(2), Req(3), 1, GTparam.Dates.DTnum);
        GuidingCentre.L = norm(Req)/Units.RE2m;

        GuidingCentre.Rline = Rline;
        GuidingCentre.Bline = Bline;
    end

    % Save to GTmag
    GTmag.Invariants = Invariants;
    GTmag.PitchAngles = PitchAngles;
    GTmag.MirrorPoints = MirrorPoints;
    if exist('Lshell', 'var')
        GTmag.Lshell = Lshell;
%        GTmag.Lgen = Lgen;
    else
        GTmag.Lshell = [];
%        GTmag.Lgen = [];
    end
    if exist('GuidingCentre', 'var')
        GTmag.GuidingCentre = GuidingCentre;
    else
        GTmag.GuidingCentre = [];
    end
end

%% Get Mirror Point function
function [num_mirror, num_eq_pitchS, HmS, HeqS, num_B0] = GetMirrorPoint(pitch, H)
    a = pitch(2:length(pitch)) - 90;
    a = find((pitch(1:length(pitch)-1) - 90) .* a < 0);
    pitch_bound_tol = 0.4;
    n = 1;
    for i = 1 : length(a)-1
        if max(abs(pitch(a(n):a(n+1)) - 90)) < pitch_bound_tol
            a(n+1) = [];
        else
            n = n + 1;
        end
    end
    num_mirror = zeros(1, length(a));
    num_eq_pitchS = zeros(1, length(a)-1);
    num_B0 = zeros(1, length(a)-1);
    if ~isempty(a) 
        num_eq_1 = 1;
        for i = 1 : length(a)-1
            b = find((abs(pitch(a(i):a(i+1)) - 90)) == max(abs(pitch(a(i):a(i+1)) - 90)));
            num_eq_2 = a(i) + b - 1;
            d = find(H(num_eq_1:num_eq_2) == max(H(num_eq_1:num_eq_2)));
            num_mirror(i) = num_eq_1 + d - 1;
            num_eq_1 = num_eq_2;
            num_eq_pitchS(i) = num_eq_2;
            num_B0(i) = a(i) + find(H(a(i):a(i+1)) == min(H(a(i):a(i+1))));
        end
        d = find(H(num_eq_1:end) == max(H(num_eq_1:end)));
        num_mirror(end) = num_eq_1 + d - 1;
    end
    HmS = H(num_mirror);
    HeqS = H(num_eq_pitchS);
    
    if isempty(num_mirror)
        num_mirror = 0;
    end
    if isempty(num_eq_pitchS)
        num_eq_pitchS = 0;
    end
    if isempty(num_B0)
        num_B0 = 0;
    end
end

function I2 = GetSecondInvariant(X, Y, Z, num_mirror, Hx, Hy, Hz, Hn, HmS)
    I2 = zeros(1, length(num_mirror) - 1);
    I2_tol = 0.2;
    for i = 1:length(num_mirror)-1
        Hm = max(HmS(i), HmS(i+1));
        H_coil = Hn(num_mirror(i):num_mirror(i+1));
        b = [X(num_mirror(i)+1:num_mirror(i+1)+1); Y(num_mirror(i)+1:num_mirror(i+1)+1); Z(num_mirror(i)+1:num_mirror(i+1)+1)];
        S = b - [X(num_mirror(i):num_mirror(i+1)); Y(num_mirror(i):num_mirror(i+1)); Z(num_mirror(i):num_mirror(i+1))];
        I2(i) = sum(sqrt(1 - H_coil/Hm).*abs((S(1,:).*Hx(num_mirror(i):num_mirror(i+1)) + S(2,:).*Hy(num_mirror(i):num_mirror(i+1)) + S(3,:).*Hz(num_mirror(i):num_mirror(i+1)))./H_coil));
    end
    I2 = I2(I2 > I2_tol * max(I2));
end


function Lshell = GetLshell(I2, BmS)
    RE = 6378137.1; % m
    k0 = 31100*1e-5; % Gauss * RE^3
    k0 = k0*1e-4*RE^3; % Tesla * m^3
    %E = [2.0e-4 1.0e-6 7.0e-8 6.0e-7 1.2e-5 5.0e-4];
    a = [ 3.0062102e-1  6.2337691e-1  6.228644e-1   6.222355e-1    2.0007187  -3.0460681;
          3.33338e-1    4.3432642e-1  4.3352788e-1  4.3510529e-1  -1.8461796e-1        1;
          0             1.5017245e-2  1.4492441e-2  1.2817956e-2   1.2038224e-1        0;
          0             1.3714667e-3  1.1784234e-3  2.1680398e-3  -6.7310339e-3        0;
          0             8.2711096e-5  3.8379917e-5 -3.2077032e-4   2.170224e-4         0;
          0             3.2916354e-6 -3.3408822e-6  7.9451313e-5  -3.8049276-6         0;
          0             8.1048663e-8 -5.3977642e-7 -1.2531932e-5   2.8212095e-8        0;
          0             1.0066362e-9 -2.1997983e-8  9.9766148e-7   0                   0;
          0             8.3232531e-13 2.3028767e-9 -3.958306e-8    0                   0;
          0            -8.1537735e-14 2.6047023e-10 6.3271665e-10  0                   0 ];
    I = sum(I2)/length(I2);
    Bm = sum(BmS)/length(BmS);
    X = log(I^3 * Bm / k0);
    an = a(:, 1)*(X < -22) + a(:, 2)*(-22 < X && X < -3) + a(:, 3)*(-3 < X && X < 3) ...
        + a(:, 4)*(3 < X && X < 11.7) + a(:, 5)*(11.7 < X && X < 23) + a(:, 6)*(X > 23);
    Y = sum(an'.*X.^(0:9));
    %dY = Y * (E(1)*(X < -22) + E(2)*(-22 < X && X < -3) + E(3)*(-3 < X && X < 3) ...
    %    + E(4)*(3 < X && X < 11.7) + E(5)*(11.7 < X && X < 23) + E(6)*(X > 23)); %вообще, еще ошибка I и Bm,сидящих в X
    Lshell = (k0 / RE^3 / Bm * (1 + exp(Y))) ^ (1/3);
    % dL = ...
end