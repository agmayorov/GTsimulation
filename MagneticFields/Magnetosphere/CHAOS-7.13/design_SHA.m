function [A_r, A_theta, A_phi] = design_SHA(r, theta, phi, N, varargin);
% [A_r, A_theta, A_phi] = design_SHA(r, theta, phi, N)
%
% Calculates design matrices A_i that connects the vector 
% of (Schmidt-normalized) spherical harmonic expansion coefficients, 
% x = (g_1^0; g_1^1; h_1^1; g_2^0; g_2^1; h_2^1; ... g_N^N; h_N^N) 
% and the magnetic component B_i, where "i" is "r", "theta" or "phi":
%        B_i = A_i*x
% Input: r(:)      radius vector (in units of the reference radius a)
%        theta(:)  colatitude    (in radians)
%        phi(:)    longitude     (in radians)
%        N         maximum degree/order
%
% [A_r, A_theta, A_phi] = design_SHA(r, theta, phi, N, i_e_flag)
% with i_e_flag = 'int' for internal sources (g_n^m and h_n^m)
%                 'ext' for external sources (q_n^m and s_n^m) 
%
% Uses MEX file design_SHA_m if available, and Matlab program design_SHA_matlab.m else

% January 2003, Nils Olsen, DSRI

% determine size of input arrays
max_size = max([size(r); size(theta); size(phi)]); 
max_length = max_size(1)*max_size(2);
% convert to matrix if input parameter is scalar
if length(r)     == 1; r = r*ones(max_size); end;
if length(theta) == 1; theta = theta*ones(max_size); end;
if length(phi)   == 1; phi = phi*ones(max_size); end;
% check for equal length of all input parameters
if size(r) ~= size(theta) | size(r) ~= size(phi);
    error('Variables must be of equal size (or scalars)');
    return
end
% convert to row vector
r = reshape(r, max_length, 1);
theta = reshape(theta, max_length, 1);
phi = reshape(phi, max_length, 1);

if nargin == 4; 
   i_e_flag = 'int'; % internal sources by default
elseif nargin > 4
   i_e_flag = varargin{1};
else
   error('At least 4 inputs required')
end; 

if exist('design_SHA_m') == 3
    [A_r, A_theta, A_phi] = design_SHA_m(r, theta, phi, N, i_e_flag);
else
    [A_r, A_theta, A_phi] = design_SHA_matlab(r, theta, phi, N, i_e_flag);
end

% -----------------------------------------------------------------------

function [A_r, A_theta, A_phi] = design_SHA_matlab(r, theta, phi, N, varargin);
% [A_r, A_theta, A_phi] = design_SHA_matlab(r, theta, phi, N)
%
% Calculates design matrices A_i that connects the vector 
% of (Schmidt-normalized) spherical harmonic expansion coefficients, 
% x = (g_1^0; g_1^1; h_1^1; g_2^0; g_2^1; h_2^1; ... g_N^N; h_N^N) 
% and the magnetic component B_i, where "i" is "r", "theta" or "phi":
%        B_i = A_i*x
% Input: r(:)      radius vector (in units of the reference radius a)
%        theta(:)  colatitude    (in radians)
%        phi(:)    longitude     (in radians)
%        N         maximum degree/order
%
% [A_r, A_theta, A_phi] = design_SHA(r, theta, phi, N, i_e_flag)
% with i_e_flag = 'int' for internal sources (g_n^m and h_n^m)
%                 'ext' for external sources (q_n^m and s_n^m) 
%

% March 2001, Nils Olsen, DSRI

if nargin == 4; 
   i_e_flag = 'int'; % internal sources by default
elseif nargin > 4
   i_e_flag = varargin{1};
else
   error('At least 4 inputs required')
end; 
N_koeff=(N+1)^2-1;

cos_theta = cos(theta);
sin_theta = sin(theta);
N_data    = length(theta);

A_r       = zeros(N_data, N_koeff);
A_theta   = zeros(N_data, N_koeff);
A_phi     = zeros(N_data, N_koeff);

k=0;
for n = 1:N
   if strcmp(i_e_flag, 'int')
      r_n       = r.^(-(n+2));
   elseif strcmp(i_e_flag, 'ext')
      r_n       = r.^(n-1);
   else
      warning 'i_e_flag neither "int" nor "ext". Assumed "int".'
      r_n       = r.^(-(n+2));
   end
   
   Pnm = legendre(n, cos_theta, 'sch')';      % P_n^m and derivatives vrt. theta
   dPnm(:,n+1) =  (sqrt(n/2).*Pnm(:,n));      % m=n
   dPnm(:,1) = -sqrt(n*(n+1)/2.).*Pnm(:,2);   % m=0
   if n > 1; dPnm(:,2)=(sqrt(2*(n+1)*n).*Pnm(:,1)-sqrt((n+2)*(n-1)).*Pnm(:,3))/2; end; % m=1
   for m = 2:n-1                              % m=2...n-1
       dPnm(:,m+1)=(sqrt((n+m)*(n-m+1)).*Pnm(:,m)-sqrt((n+m+1)*(n-m)).*Pnm(:,m+2))/2;
   end;
   if n == 1 dPnm(:,2) = sqrt(2)*dPnm(:,2); end
   
   if ~strcmp(i_e_flag, 'ext') % internal sources by default
      for m = 0:n;  
         cos_phi   = cos(m*phi);
         sin_phi   = sin(m*phi);
         
         if m == 0
            k = k+1;       % terms corresponding to g_n^0
            A_r(:,k)       =  (n+1).*r_n(:).*Pnm(:,1);
            A_theta(:,k)   = -r_n(:).*dPnm(:,1);
            A_phi(:,k)     =  r_n(:)*0;
         else
            k = k+1;       % terms corresponding to g_n^m
            A_r(:,k)       =  (n+1).*r_n(:).*cos_phi.*Pnm(:,m+1);
            A_theta(:,k)   = -r_n(:).*cos_phi.*dPnm(:,m+1);
            A_phi(:,k)     =  r_n(:).*m.*sin_phi.*Pnm(:,m+1)./sin_theta;
            
            k = k+1;       % terms corresponding to h_n^m
            A_r(:,k)       =  (n+1).*r_n(:).*sin_phi.*Pnm(:,m+1);
            A_theta(:,k)   = -r_n(:).*sin_phi.*dPnm(:,m+1);
            A_phi(:,k)     = -r_n(:).*m.*cos_phi.*Pnm(:,m+1)./sin_theta;
         end    
      end; % m
   else % external sources, i_e_flag ='ext'
      for m = 0:n;  
         cos_phi   = cos(m*phi);
         sin_phi   = sin(m*phi);
         
         if m == 0
            k = k+1;       % terms corresponding to q_n^0
            A_r(:,k)       = -n.*r_n(:).*Pnm(:,1);
            A_theta(:,k)   = -r_n(:).*dPnm(:,1);
            A_phi(:,k)     =  r_n(:)*0;
         else
            k = k+1;       % terms corresponding to q_n^m
            A_r(:,k)       = -n.*r_n(:).*cos_phi.*Pnm(:,m+1);
            A_theta(:,k)   = -r_n(:).*cos_phi.*dPnm(:,m+1);
            A_phi(:,k)     =  r_n(:).*m.*sin_phi.*Pnm(:,m+1)./sin_theta;
            
            k = k+1;       % terms corresponding to s_n^m
            A_r(:,k)       = -n.*r_n(:).*sin_phi.*Pnm(:,m+1);
            A_theta(:,k)   = -r_n(:).*sin_phi.*dPnm(:,m+1);
            A_phi(:,k)     = -r_n(:).*m.*cos_phi.*Pnm(:,m+1)./sin_theta;
         end    
      end; % m
   end; % if int/ext
end; % n

