function B_ext = synth_values_CHAOS_ext(t, r, theta, phi, RC_ei, model_ext, varargin)
% B_ext = synth_values_CHAOS_ext(t, r, theta, phi, RC_ei, model_ext)
% B_ext = synth_values_CHAOS_ext(t, r, theta, phi, RC_ei, model_ext, dipole, filename_gsm)
%
% Input
% -----
% t : Nx1
%   Time as column vector of length N (MD2000 in days).
% r : Nx1
%   Radius column vector (km).
% theta : Nx1
%   Geographic colatitude (in degrees).
% phi : Nx1
%   Geographic longitude (in degrees).
% RC_ei : Nx2
%   RC index, 2 columns: internal, external.
% model_ext: struct
%   External model coefficients.
% dipole : 1x3, optional (but must be given with filename_gsm)
%   IGRF dipole coefficients: [g10, g11, h11].
% filename_gsm : str, optional (but must be given with dipole)
%   Path to GSM/SM coefficient mat-file.
%
% Output
% ------
% B_ext : Nx3
%   Magnetic field [B_r(:), B_theta(:), B_phi(:)] in nT
%
% March 2019, Clemens Kloss, DTU Space
% October 2014, Chris Finlay and Nils Olsen, DTU Space

rad = pi/180;
a = 6371.2;

if nargin==8 
   dipole = varargin{1};
   filename_gsm = varargin{2};
elseif nargin==6
   % do nothing
else
   error('Wrong number of input arguments: only 6 or 8 possible.');
end

% calculate using code from create_G_spline_2.m
N_data = length(t);
N_t_segments_1 = length(model_ext.t_break_q10)-1; % breaks for q10
N_t_segments_2 = length(model_ext.t_break_qs11)-1; % breaks for q11, s11

N_Data_Frac = 200;

B_ext = zeros(N_data, 3);
for i=1:N_Data_Frac:N_data
    index = [i:min(i+N_Data_Frac-1, N_data)]';  %take [i:i+N_Data_Frac-1] but maximum up to N_Data
    
    if nargin==8
        [theta_sm, phi_sm, R_1]   = geo2sm(t(index), theta(index), phi(index), dipole);
    else  % pass defaults handling to geo2sm
        [theta_sm, phi_sm, R_1]   = geo2sm(t(index), theta(index), phi(index));
    end
    [A_r_ext, A_theta_sm_ext, A_phi_sm_ext]         = design_SHA(r(index)/a, theta_sm*rad, phi_sm*rad, 2, 'ext');
    [A_r_ind, A_theta_sm_ind, A_phi_sm_ind] = design_SHA(r(index)/a, theta_sm*rad, phi_sm*rad, 2, 'int');
    N_coeff_sm = size(A_r_ext, 2);
    
    A_theta_ext = repmat(R_1(:,1,1), 1, N_coeff_sm).*A_theta_sm_ext + repmat(R_1(:,2,1), 1, N_coeff_sm).*A_phi_sm_ext;
    A_phi_ext =   repmat(R_1(:,1,2), 1, N_coeff_sm).*A_theta_sm_ext + repmat(R_1(:,2,2), 1, N_coeff_sm).*A_phi_sm_ext;
    A_theta_ind = repmat(R_1(:,1,1), 1, N_coeff_sm).*A_theta_sm_ind + repmat(R_1(:,2,1), 1, N_coeff_sm).*A_phi_sm_ind;
    A_phi_ind =   repmat(R_1(:,1,2), 1, N_coeff_sm).*A_theta_sm_ind + repmat(R_1(:,2,2), 1, N_coeff_sm).*A_phi_sm_ind;
    
    if nargin==8
        [A_r_sm, A_theta_sm, A_phi_sm]     = design_SHA_sm(t(index), r(index)/a, theta(index)*rad, phi(index)*rad, [1 1 1 1 1 1 1 1], filename_gsm);
        [A_r_gsm, A_theta_gsm, A_phi_gsm]  = design_SHA_gsm(t(index), r(index)/a, theta(index)*rad, phi(index)*rad, [1 1 1 1], filename_gsm);
    else  % pass defaults handling to desgin_SHA_gsm/sm
        [A_r_sm, A_theta_sm, A_phi_sm]     = design_SHA_sm(t(index), r(index)/a, theta(index)*rad, phi(index)*rad, [1 1 1 1 1 1 1 1]);
        [A_r_gsm, A_theta_gsm, A_phi_gsm]  = design_SHA_gsm(t(index), r(index)/a, theta(index)*rad, phi(index)*rad, [1 1 1 1]);
    end        
    A_r_sm_gsm = [A_r_sm A_r_gsm(:, [1 4])];
    A_theta_sm_gsm = [A_theta_sm A_theta_gsm(:, [1 4])];
    A_phi_sm_gsm = [A_phi_sm A_phi_gsm(:, [1 4])];
    
    % add the terms that change explicitely in time (q10, ...)
    A_break_1 = zeros(length(index), N_t_segments_1);
    [~, whichBin] = histc(t(index), model_ext.t_break_q10);
    for it = 1:N_t_segments_1
        A_break_1(whichBin == it, it) = 1;
    end
    A_break_2 = zeros(length(index), N_t_segments_2);
    [~, whichBin] = histc(t(index), model_ext.t_break_qs11);
    for it = 1:N_t_segments_2
        A_break_2(whichBin == it, it) = 1;
    end
    
    % q10, q11, s11
    A_phi_q10   = [repmat(A_phi_sm_gsm(:,1),   1, N_t_segments_1).*A_break_1 ...
        repmat(A_phi_sm_gsm(:,2),   1, N_t_segments_2).*A_break_2 ...
        repmat(A_phi_sm_gsm(:,3),   1, N_t_segments_2).*A_break_2];
    A_theta_q10 = [repmat(A_theta_sm_gsm(:,1), 1, N_t_segments_1).*A_break_1 ...
        repmat(A_theta_sm_gsm(:,2), 1, N_t_segments_2).*A_break_2 ...
        repmat(A_theta_sm_gsm(:,3), 1, N_t_segments_2).*A_break_2];
    A_r_q10     = [repmat(A_r_sm_gsm(:,1),     1, N_t_segments_1).*A_break_1 ...
        repmat(A_r_sm_gsm(:,2),     1, N_t_segments_2).*A_break_2 ...
        repmat(A_r_sm_gsm(:,3),     1, N_t_segments_2).*A_break_2];
    
    A_r = [A_r_q10 A_r_sm_gsm(:,4:end) ...
        RC_ei(index,1).*A_r_ext(:,1) + RC_ei(index,2).*A_r_ind(:,1) ...
        RC_ei(index,1).*A_r_ext(:,2) + RC_ei(index,2).*A_r_ind(:,2) ...
        RC_ei(index,1).*A_r_ext(:,3) + RC_ei(index,2).*A_r_ind(:,3) ...
        ];
    A_theta = [A_theta_q10 A_theta_sm_gsm(:,4:end) ...
        RC_ei(index,1).*A_theta_ext(:,1) + RC_ei(index,2).*A_theta_ind(:,1) ...
        RC_ei(index,1).*A_theta_ext(:,2) + RC_ei(index,2).*A_theta_ind(:,2) ...
        RC_ei(index,1).*A_theta_ext(:,3) + RC_ei(index,2).*A_theta_ind(:,3) ...
        ];
    A_phi = [A_phi_q10 A_phi_sm_gsm(:,4:end) ...
        RC_ei(index,1).*A_phi_ext(:,1) + RC_ei(index,2).*A_phi_ind(:,1) ...
        RC_ei(index,1).*A_phi_ext(:,2) + RC_ei(index,2).*A_phi_ind(:,2) ...
        RC_ei(index,1).*A_phi_ext(:,3) + RC_ei(index,2).*A_phi_ind(:,3) ...
        ];
    
    B_r = A_r*[model_ext.q10; model_ext.qs11(:); model_ext.m_sm(4:end); model_ext.m_gsm; model_ext.m_Dst];
    B_theta = A_theta*[model_ext.q10; model_ext.qs11(:); model_ext.m_sm(4:end); model_ext.m_gsm; model_ext.m_Dst];
    B_phi = A_phi*[model_ext.q10; model_ext.qs11(:); model_ext.m_sm(4:end); model_ext.m_gsm; model_ext.m_Dst];
    B_ext(index,:) = [B_r B_theta B_phi];
end

