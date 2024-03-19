function [theta_sm, phi_sm, varargout] = geo2sm(t, theta_geo, phi_geo, varargin)
% [theta_sm, phi_sm] = geo2gsm(t, theta_geo, phi_geo);
% [theta_sm, phi_sm, R] = geo2gsm(t, theta_geo, phi_geo);
% [theta_sm, phi_sm, R] = geo2gsm(t, theta_geo, phi_geo, [theta_b phi_b]);
% [theta_sm, phi_sm, R] = geo2gsm(t, theta_geo, phi_geo, [g_1^0 g_1^1 h_1^1]);
%
% Input:  geographic theta_geo,  phi_geo  (in degrees)
% Output: theta_sm, phi_sm (in degrees)
%         R(:,2,2) is the matrix that rotates a horizontal vector from GEO to SM:
%                  [B_theta_sm; B_phi_sm] = R*[B_theta_geo; B_phi_geo]
%
% September 2004, Nils Olsen, DSRI

rad = pi/180;

% determine size of input arrays
max_size = max([size(t); size(theta_geo); size(phi_geo)]); 
max_length = max_size(1)*max_size(2);
% convert to matrix if input parameter is scalar
if length(t)         == 1; t = repmat(t, max_size); end;
if length(theta_geo) == 1; theta_geo = repmat(theta_geo, max_size); end;
if length(phi_geo)   == 1; phi_geo = repmat(phi_geo, max_size); end;
% check for equal length of all input parameters
if length(t) ~= length(theta_geo) || length(t) ~= length(phi_geo);
    error('Variables must be of equal size (or scalars)');
    return
end

% convert to row vectors
t = reshape(t, max_length, 1);
theta_geo = reshape(theta_geo, max_length, 1);
phi_geo = reshape(phi_geo, max_length, 1);

% Initialization: coordinates of geomagnetic North pole
if nargin > 3
    tmp = varargin{1};
    if length(tmp) == 2; % coordinates theta_b and phi_b of dipole N-pole
        theta_b = tmp(1);
        phi_b = tmp(2);
    elseif length(tmp) == 3; % Gauss coefficients g10, g11, h11
        phi_b = atan2(-tmp(3), -tmp(2))/rad;
        theta_b = atan2(sqrt(tmp(2).^2 + tmp(3).^2), -tmp(1))/rad;
    else
        error('wrong input of input argument #6: neither [theta_b phi_b] nor [g_1^0 g_1^1 h_1^1]');
    end
else
    warning('Missing coordinates of geomagnetic North pole. Falling back to IGRF-12 2015.');
    theta_b = 9.69; phi_b = 287.37;  % IGRF-12 2015
end

% GEO spherical -> GEO cartesian coordinates
c_t = cos(theta_geo*rad); s_t = sin(theta_geo*rad);
c_p = cos(phi_geo*rad);   s_p = sin(phi_geo*rad);
z_geo = c_t;
x_geo = s_t .* c_p;
y_geo = s_t .* s_p;

[S, GST] = sunGEI(t);

% transformation GEO -> GEI
A(:,1,1) = cos(GST);  A(:,1,2) = -sin(GST);  A(:,1,3) = 0; 
A(:,2,1) = sin(GST);  A(:,2,2) =  cos(GST);  A(:,2,3) = 0; 
A(:,3,1) = 0;         A(:,3,2) = 0;          A(:,3,3) = 1;

x_gei = A(:,1,1).*x_geo + A(:,1,2).*y_geo;
y_gei = A(:,2,1).*x_geo + A(:,2,2).*y_geo;
z_gei = z_geo;

% transformation GEI -> SM
Dgeo = [sin(theta_b*rad)*cos(phi_b*rad); sin(theta_b*rad)*sin(phi_b*rad); cos(theta_b*rad)];
Dgei = zeros(length(t), 3);
Dgei(:,1) = A(:,1,1).*Dgeo(1) + A(:,1,2).*Dgeo(2);
Dgei(:,2) = A(:,2,1).*Dgeo(1) + A(:,2,2).*Dgeo(2);
Dgei(:,3) = Dgeo(3);

Y = cross(Dgei, S, 2); 
Y_norm = sqrt(Y(:,1).^2 + Y(:,2).^2 + Y(:,3).^2);
Y = [Y(:,1)./Y_norm Y(:,2)./Y_norm Y(:,3)./Y_norm]; 

X = cross(Y, Dgei, 2); 

M = zeros(length(t), 3, 3);
M(:,1,:) = X;
M(:,2,:) = Y;
M(:,3,:) = Dgei;

x_sm = M(:,1,1).*x_gei + M(:,1,2).*y_gei + M(:,1,3).*z_gei;
y_sm = M(:,2,1).*x_gei + M(:,2,2).*y_gei + M(:,2,3).*z_gei;
z_sm = M(:,3,1).*x_gei + M(:,3,2).*y_gei + M(:,3,3).*z_gei;

% SM cartesian -> SM spherical coordinates
theta_sm = 90 - atan2(z_sm, sqrt(x_sm.^2 + y_sm.^2))/rad;
phi_sm = mod(atan2(y_sm, x_sm)/rad, 360);

if nargout == 3 % also calculate matrix R_3 that transform GEO -> SM
    % transformation from GEO/spherical -> GEO/cartesian
    c_t = cos(theta_geo*rad); s_t = sin(theta_geo*rad);
    c_p = cos(phi_geo*rad);   s_p = sin(phi_geo*rad); 
    R_1 = zeros(size(c_p, 1), 3, 3);
    R_1(:,1,:) = [s_t.*c_p  c_t.*c_p  -s_p];
    R_1(:,2,:) = [s_t.*s_p  c_t.*s_p  +c_p];
    R_1(:,3,:) = [c_t       -s_t   zeros(size(c_p))];
    % transformation from SM/cartesian -> SM/spherical
    c_t = cos(theta_sm*rad); s_t = sin(theta_sm*rad);
    c_p = cos(phi_sm*rad);   s_p = sin(phi_sm*rad); 
    R_2 = zeros(size(c_p, 1), 3, 3);
    R_2(:,1,:) = [s_t.*c_p  s_t.*s_p  +c_t];
    R_2(:,2,:) = [c_t.*c_p  c_t.*s_p  -s_t];
    R_2(:,3,:) = [-s_p      c_p      zeros(size(c_p))];
    % R_3 is transformation matrix from SM/spherical -> GEO/spherical
    R = mat_mul_mat(M,A);    % GEO/cartesian -> SM/cartesian
    R_tmp = mat_mul_mat(R, R_1);
    R_3 = mat_mul_mat(R_2, R_tmp);
    varargout{1} = R_3(:, 2:3, 2:3);
end
