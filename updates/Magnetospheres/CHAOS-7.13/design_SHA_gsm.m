function [A_r_gsm, A_theta_gsm, A_phi_gsm] = design_SHA_gsm(t, r, theta_geo, phi_geo, varargin);
%
% [A_r_gsm, A_theta_gsm, A_phi_gsm] = design_SHA_gsm(t, r, theta_geo, phi_geo);
% [A_r_gsm, A_theta_gsm, A_phi_gsm] = design_SHA_gsm(t, r, theta_geo, phi_geo, coeffs);
% [A_r_gsm, A_theta_gsm, A_phi_gsm] = design_SHA_gsm(t, r, theta_geo, phi_geo, coeffs, filename_gsm);
% [A_r_gsm, A_theta_gsm, A_phi_gsm] = design_SHA_gsm(t, r, theta_geo, phi_geo, coeffs, filename_gsm, i_e_flag);
%
% Calculates design matrices A_i that connects the vector
% of (Schmidt-normalized) spherical harmonic expansion coefficients,
% x = (g_1^0; g_1^1; h_1^1; g_2^0; g_2^1; h_2^1; ... g_N^N; h_N^N)
% for a magnetic field in the GSM frame
% The magnetic component B_i, where "i" is "r", "theta" or "phi":
%        B_i = A_i*x
% Input: t(:)         time (MD2000 in days)
%        r(:)         radius vector (in units of the reference radius a)
%        theta_geo(:) geographic colatitude    (in radians)
%        phi_geo(:)   geographic longitude     (in radians)
% Optional: coeffs    array indicating which coefficients to be included
%                     E.g.: coeffs = 1:         only q_10_GSM
%                           coeffs = [1 0 1]:   q_10_GSM, s_11_GSM
%                           coeffs = [1 0 0 1]: q_10_GSM, q_20_GSM
%                           (default: 1, i.e., only q_10_GSM)
%           filename_gsm    Name of file with SM/GSM expansion coefficients
%                           (default: "SM_GSM_SHA.mat" in same directory as
%                           this function)
%           i_e_flag        Flag indicating whether external (e) and/or
%                           induced (i) contributions are computed.
%                           (default: "e+i")
%
% August 2006, Nils Olsen, DNSC
%

global SM_M_sorted SM_omega_sorted SM_M_sorted_ind SM_omega_sorted_ind ...
    GSM_M_sorted GSM_omega_sorted GSM_M_sorted_ind GSM_omega_sorted_ind


% The coefficient file is assumed to be in the same directory as this
% function.
[st, i] = dbstack('-completenames');
GSM_path = st(1).file(1:strfind(st(1).file, st(1).name)-1);

if nargin > 4 & ~isempty(varargin{1})
    coeffs = varargin{1};
else
    coeffs = 1;
end

if nargin > 5 & ~isempty(varargin{2})
    filename_gsm = varargin{2};
else
    filename_gsm = [GSM_path 'GSM_SM_SHA.mat'];
end

if nargin > 6  & ~isempty(varargin{3})
    i_e_flag = lower(varargin{3});  % convert to lowercase letters
else
    i_e_flag = 'i+e';
end

N = 20;
if length(GSM_omega_sorted) ~= N
    if exist(filename_gsm, 'file') > 0
        load(filename_gsm)
    else
        error(['file ' filename_gsm ' not found']);
    end
end

% synthesis
q_GSM = zeros(length(coeffs), size(GSM_M_sorted, 2), length(t)); g_GSM = q_GSM;
for k = 1:length(coeffs)
    if coeffs(k) ~= 0
        y = zeros(24, length(t)); y_ind = y;
        if k < 4; l_max = 3; else; l_max = 24; end;
        for l = 1:l_max
            n = 1;
            while abs(GSM_M_sorted(k,l,n)) > 1e-6 & n < N
                y(l,:) = y(l,:) + squeeze(real(GSM_M_sorted(k,l,n)*exp(-j*GSM_omega_sorted(k,l,n)*t))');
                n = n+1;
            end
            n = 1;
            while abs(GSM_M_sorted_ind(k,l,n)) > 1e-6 & n < N
                y_ind(l,:) = y_ind(l,:) + squeeze(real(GSM_M_sorted_ind(k,l,n)*exp(-j*GSM_omega_sorted_ind(k,l,n)*t))');
                n = n+1;
            end
            %          for n = 1:N
            %              y(l,:) = y(l,:) + squeeze(real(GSM_M_sorted(k,l,n)*exp(-j*GSM_omega_sorted(k,l,n)*t))');
            %              y_ind(l,:) = y_ind(l,:) + squeeze(real(GSM_M_sorted_ind(k,l,n)*exp(-j*GSM_omega_sorted_ind(k,l,n)*t))');
            %          end
        end
        q_GSM(k, :, :) = y;
        g_GSM(k, :, :) = y_ind;
    end
end

A_r_gsm = zeros(length(t), length(coeffs));
A_theta_gsm = zeros(length(t), length(coeffs));
A_phi_gsm = zeros(length(t), length(coeffs));

% external fields ...
if strfind(i_e_flag, 'e') > 0
    [A_r_geo, A_theta_geo, A_phi_geo] = design_SHA(r, theta_geo, phi_geo, floor(sqrt(length(coeffs))), 'ext');
    for k = 1:length(coeffs)
        if coeffs(k) ~= 0
            nn = floor(sqrt(k));  % degree that corresponds to current index
            n_start = nn^2;  % start index
            n_stop = n_start + 2*nn;  % stop index
            for n = n_start:n_stop
                A_r_gsm(:,k)     = A_r_gsm(:,k)     + squeeze(q_GSM(k,n,:)).*A_r_geo(:,n);
                A_theta_gsm(:,k) = A_theta_gsm(:,k) + squeeze(q_GSM(k,n,:)).*A_theta_geo(:,n);
                A_phi_gsm(:,k)   = A_phi_gsm(:,k)   + squeeze(q_GSM(k,n,:)).*A_phi_geo(:,n);
            end
        end
    end
end

% ... induced fields
if strfind(i_e_flag, 'i') > 0
    [A_r_geo, A_theta_geo, A_phi_geo] = design_SHA(r, theta_geo, phi_geo, floor(sqrt(length(coeffs))), 'int');
    for k = 1:length(coeffs)
        if coeffs(k) ~= 0
            nn = floor(sqrt(k));  % degree that corresponds to current index
            n_start = nn^2;  % start index
            n_stop = n_start + 2*nn;  % stop index
            for n = n_start:n_stop
                A_r_gsm(:,k)     = A_r_gsm(:,k)     + squeeze(g_GSM(k,n,:)).*A_r_geo(:,n);
                A_theta_gsm(:,k) = A_theta_gsm(:,k) + squeeze(g_GSM(k,n,:)).*A_theta_geo(:,n);
                A_phi_gsm(:,k)   = A_phi_gsm(:,k)   + squeeze(g_GSM(k,n,:)).*A_phi_geo(:,n);
            end
        end
    end
end

