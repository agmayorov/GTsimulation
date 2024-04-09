function [Bx_tot, By_tot, Bz_tot] = JF12mod(X, Y, Z)
%   [Bx_tot, By_tot, Bz_tot] = JF12mod(X, Y, Z)
%   The JF12 model of the magnetic field in the Galaxy (doi: 10.48550/arXiv.1204.3662).
%   Ver. 2, red. 2 / March 2023 / A. Kirichenko, A. Mayorov, R. Yulbarisov / CRTeam / NRNU MEPhI, Russia
%
%   Arguments:
%       X, Y, Z                    - Float     -   Position, kpc
%   Output:
%       Bx_tot, By_tot, Bz_tot     - Float     -   Magnetic field components, nT

    [phi, r, ~] = cart2pol(X, Y, Z);
    R = sqrt(X^2 + Y^2 + Z^2);

    % ===========================
    % REGULAR (large-scale) field
    % ===========================

    persistent b_disk r_arms b_ring h_disk w_disk
    persistent pitch sinPitch cosPitch
    persistent B_n B_s r_n r_s w_h z_0
    persistent bX thetaX0 sinThetaX0 cosThetaX0 tanThetaX0 rXc rX

    if isempty(b_disk)
        % Disk
        b_disk = [0.1, 3.0, -0.9, -0.8, -2.0, -4.2,  0.0,  2.7];
        r_arms = [5.1, 6.3,  7.1,  8.3,  9.8, 11.4, 12.7, 15.5];
        b_ring = 0.1;
        h_disk = 0.4;
        w_disk = 0.27;

        pitch = 11.5 * pi/180;
        sinPitch = sin(pitch);
        cosPitch = cos(pitch);

        % Toroidal halo
        B_n =  1.4;
        B_s = -1.1;
        r_n = 9.22;
        r_s = 16.7;
        w_h = 0.2;
        z_0 = 5.3;

        % X halo
        bX = 4.6;
        thetaX0 = 49.0 * pi/180;
        sinThetaX0 = sin(thetaX0);
        cosThetaX0 = cos(thetaX0);
        tanThetaX0 = tan(thetaX0);
        rXc = 4.8;
        rX = 2.9;
    end

    % Disk component
    Bx_d = 0;
    By_d = 0;
    if r >= 3 && r < 5
        % Molecular ring
        lfDisk = logisticFunction(Z, h_disk, w_disk);
        bMag = b_ring * (5/r) * (1 - lfDisk);
        Bx_d = -bMag * sin(phi);
        By_d =  bMag * cos(phi);
    elseif r >= 5 && r <= 20
        % Spiral arms
        lfDisk = logisticFunction(Z, h_disk, w_disk);
        plus2pi = 0;
        r_negx = r * exp(-(phi - pi + plus2pi) / tan(pi/2 - pitch));
        while r_negx > r_arms(8)
            plus2pi = plus2pi + 2*pi;
            r_negx = r * exp(-(phi - pi + plus2pi) / tan(pi/2 - pitch));
        end
        bMag = b_disk( find(r_negx <= r_arms, 1) );
        bMag = bMag * (5/r) * (1 - lfDisk);
        Bx_d = bMag * (sinPitch*cos(phi) - cosPitch*sin(phi));
        By_d = bMag * (sinPitch*sin(phi) + cosPitch*cos(phi));
    end

    % Toroidal component
    Bx_t = 0;
    By_t = 0;
    if R >= 1 && r <= 20
        lfDisk = logisticFunction(Z, h_disk, w_disk);
        bMagH = exp(-abs(Z)/z_0) * lfDisk;
        if Z >= 0
            bMagH = bMagH * B_n * (1 - logisticFunction(r, r_n, w_h));
        else
            bMagH = bMagH * B_s * (1 - logisticFunction(r, r_s, w_h));
        end
        Bx_t = -bMagH * sin(phi);
        By_t =  bMagH * cos(phi);
    end

    % X-field component
    Bx_x = 0;
    By_x = 0;
    Bz_x = 0;
    if R >= 1 && r <= 20
        rc = rXc + abs(Z)/tanThetaX0;
        if r < rc && r > 0
            % Varying elevation region
            rp = r * rXc / rc;
            bMagX = bX * exp(-rp/rX) * (rp/r)^2;
            thetaX = atan(abs(Z)/(r - rp));
            if Z == 0
                thetaX = pi/2;
            end
            sinThetaX = sin(thetaX);
            cosThetaX = cos(thetaX);
        elseif r == 0
            % Varying elevation region: singular region
            bMagX = bX * (rXc/rc)^2;
            thetaX = pi/2;
            sinThetaX = sin(thetaX);
            cosThetaX = cos(thetaX);
        else
            % Constant elevation region
            rp = r - abs(Z)/tanThetaX0;
            bMagX = bX * exp(-rp/rX) * (rp/r);
            sinThetaX = sinThetaX0;
            cosThetaX = cosThetaX0;
        end
        Bx_x = sign(Z) * bMagX * cosThetaX * cos(phi);
        By_x = sign(Z) * bMagX * cosThetaX * sin(phi);
        Bz_x = bMagX * sinThetaX;
    end

    % Total REGULAR (large-scale) field
    Bx = Bx_d + Bx_t + Bx_x;
    By = By_d + By_t + By_x;
    Bz =               Bz_x;
    Babs = sqrt(Bx^2 + By^2 + Bz^2);

    % ===========================
    %      NON-REGULAR field
    % ===========================

    persistent f_a f_i beta_str b_int_turb b_disk_turb z_disk_turb
    persistent b_halo_turb r_halo_turb z_halo_turb
    persistent Gx Gy Gz x_grid y_grid z_grid

    if isempty(f_a)
        % Rescaling of the JF12 random field strengths
        f_a = 0.6;
        f_i = 0.3;

        % Striated field parameter
        beta_str = 1.36;

        % Disk
        b_int_turb = 7.63;
        b_disk_turb = [10.81, 6.96, 9.59, 6.96, 1.96, 16.34, 37.29, 10.35];
        z_disk_turb = 0.61;

        % Halo
        b_halo_turb = 4.68;
        r_halo_turb = 10.97;
        z_halo_turb = 2.84;

        load G_nCell=250_boxSize=0.5kpc_lMin=4.0pc_lMax=500.0pc_seed=0.mat Gx Gy Gz x_grid y_grid z_grid
        % load G_nCell=500_boxSize=0.5kpc_lMin=2.0pc_lMax=500.0pc_seed=0.mat Gx Gy Gz x_grid y_grid z_grid

        Gx = double(Gx);
        Gy = double(Gy);
        Gz = double(Gz);
    end

    Bx_aniso = 0;
    By_aniso = 0;
    Bz_aniso = 0;

    Bx_iso = 0;
    By_iso = 0;
    Bz_iso = 0;

    if r < 20 && abs(Z) < 20
        % Extract the necessary components of G
        ix = find(mod(X, x_grid(end)) >= x_grid(1:end-1), 1, 'last');
        iy = find(mod(Y, x_grid(end)) >= y_grid(1:end-1), 1, 'last');
        iz = find(mod(Z, x_grid(end)) >= z_grid(1:end-1), 1, 'last');
        Gx_p = Gx(ix, iy, iz);
        Gy_p = Gy(ix, iy, iz);
        Gz_p = Gz(ix, iy, iz);

        % Anisotropic random field
        B_aniso_rms = sqrt(1.5 * beta_str) * dot([Bx By Bz], [Gx_p Gy_p Gz_p]) * Babs;
        Bx_aniso = B_aniso_rms * Bx;
        By_aniso = B_aniso_rms * By;
        Bz_aniso = B_aniso_rms * Bz;

        % Isotropic random field
        % Disk
        if r < 5
            B_iso_disk = b_int_turb;
        else
            plus2pi = 0;
            r_negx = r * exp(-(phi - pi + plus2pi) / tan(pi/2 - pitch));
            while r_negx > r_arms(8)
                plus2pi = plus2pi + 2*pi;
                r_negx = r * exp(-(phi - pi + plus2pi) / tan(pi/2 - pitch));
            end
            B_iso_disk = b_disk_turb( find(r_negx <= r_arms, 1) );
            B_iso_disk = B_iso_disk * (5/r);
        end
        B_iso_disk = B_iso_disk * exp(-0.5 * (Z / z_disk_turb)^2);
        % Halo
        B_iso_halo = b_halo_turb * exp(-r / r_halo_turb) * exp(-0.5 * (Z / z_halo_turb)^2);
        % General
        B_iso_rms = sqrt(B_iso_disk^2 + B_iso_halo^2);
        Bx_iso = B_iso_rms * Gx_p;
        By_iso = B_iso_rms * Gy_p;
        Bz_iso = B_iso_rms * Gz_p;
    end

    % ===========================
    %      TOTAL FIELD
    % ===========================

    Bx_tot = 0.1 * (Bx + f_a * Bx_aniso + f_i * Bx_iso); % 0.1 muG to nT
    By_tot = 0.1 * (By + f_a * By_aniso + f_i * By_iso);
    Bz_tot = 0.1 * (Bz + f_a * Bz_aniso + f_i * Bz_iso);
end

function lfDisk = logisticFunction(Z, h, w)
    lfDisk = 1/(1 + exp(-2*(abs(Z)-h)/w));
end
