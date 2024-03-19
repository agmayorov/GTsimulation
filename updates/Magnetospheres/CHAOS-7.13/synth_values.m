function [B_MF, varargout] = synth_values(r, theta, phi, m_MF, varargin)
%  B_MF         = synth_values(r, theta, phi, m_MF)
%  B_MF         = synth_values(r, theta, phi, m_MF, t, epoch, m_SV)
% [B_MF, B_Dst] = synth_values(r, theta, phi, m_MF, t, epoch, m_SV, Dst)
% [B_MF, B_Dst] = synth_values(r, theta, phi, m_MF, t, epoch, m_SV, Dst, m_ext_0)
% [B_MF, B_Dst] = synth_values(r, theta, phi, m_MF, t, epoch, m_SV, Dst, m_ext_0, m_ext_Dst)
% [B_MF, B_Dst] = synth_values(r, theta, phi, m_MF, t, epoch, m_SV, Dst, m_ext_0, m_ext_Dst, Q_1)
%  B_MF         = synth_values(r, theta, phi, m_pp, t)
%  B_MF         = synth_values(r, theta, phi, m_pp, t, boundary)
%
% r(:) is the geocentric radius [km]
% geographic co-latitude theta(:) and longitude phi(:) [degrees]
% m_MF(:) contains the main field Gauss coefficients [nT]
%         or spline structure (created using ppval) of Gauss coefficients
% t(:) is time as MJD2000 (days)
% epoch (year, YYYY) is the epoch of the SV model
% m_SV(:,:)      contains the secular variation Gauss coefficients of a Taylor expansion [nT/yr^k]
% Dst(:)       is the Dst index
% m_ext_0(:)   contains the external (non Dst-dependent) Gauss coefficients [nT]
%              m_ext_0(:,n) contains static (n=1), annual (n=2,3) and semi-annual (n=4,5) coefficients
% m_ext_Dst(:) contains the external, linear on the Dst index dependent, coefficients
% Q_1          is the ration of induced to external coefficients, for n=1
% boundary     'n': spline extrapolation outside time spanned by spline knots
%              'l': linear extrapolation
%              'c': constant extrapolation
%
% July 2007, Nils Olsen, DNSC

% NIO April 2003: spline representation added
% NIO October 2003: accepts SV with quadratic terms
% NIO October 9, 2003: quadratic SV computation corrected
% NIO February 6, 2006: cubic SV computation added
% NIO February 17, 2006: boundary values for spline extrapolation added
% NIO February 19, 2006: corrected calculation of g_i
% NIO July 30, 2007: bug in spline "linear extrapolation" mode fixed
% ANCKLO December 12, 2018: fixed bug in linear interpolation if exactly
%   one point is outside of range (need to treat index_left=[] as empty
%   matrix, see lines 148-155)

rad = pi/180;
a = 6371.2;

if nargin > 4; t = varargin{1}; end;

if nargin >= 9
    m_ext_0 = varargin{5};
    [N_koeff_ext, n_annual] = size(m_ext_0);
    maxdeg_ext = sqrt(N_koeff_ext+1)-1;
else
    m_ext_0 = [23.9756; 0.7279; -3.3196; 0.0595; -0.0714; 0.0145; -0.1146; 0.0926]; % OSVM
    maxdeg_ext = 2;
    n_annual = 1;
end
if nargin >= 10
    m_ext_Dst = varargin{6};
else
    m_ext_Dst = [-0.6413; -0.0089; 0.0905]; % Oersted(10b/01)
end
if nargin >= 11
    Q_1 = varargin{7};
else
    Q_1 = 0.28;
end
if isstruct(m_MF) == 0
    maxdeg_MF = sqrt(length(m_MF)+1)-1;
else
    maxdeg_MF = sqrt(m_MF.dim+1)-1; % if spline representation
end

% determine size of input arrays
if nargin == 4 % no SV
    max_size = max([size(r); size(theta); size(phi)]); 
    max_length = max_size(1)*max_size(2);
else 
    max_size = max([size(r); size(theta); size(phi); size(t)]); 
    max_length = max_size(1)*max_size(2);
end
% convert to matrix if input parameter is scalar
if length(r)       == 1; r = r*ones(max_size); end;
if length(theta)   == 1; theta = theta*ones(max_size); end;
if length(phi)     == 1; phi = phi*ones(max_size); end;
if nargin > 4 && length(t) == 1; t = t*ones(max_size); end;
% check for equal length of all input parameters
if (size(r) ~= size(theta) | size(r) ~= size(phi));
    error('Variables must be of equal size (or scalars)');
    return
end

r = r(:); theta = theta(:); phi = phi(:);
if nargin > 4; t = t(:); end;

N_data = length(r);
mem = 5e7;
N_Data_Frac = fix(mem/(8*3*maxdeg_MF*(maxdeg_MF+2)));

% check for spline representation
if nargin == 5 && isstruct(m_MF) == 0
    error('m_pp must be a spline structure');
end
if isstruct(m_MF) == 1 
    if nargin ~=5 && nargin ~=6; error('5 or 6 input arguments required for spline evaluation'); end
    if min(t) < m_MF.breaks(1) || max(t) > m_MF.breaks(end) warning('synth_values:t_outside_range', 't outside [breaks(1) ... breaks(end]'); end 
end

if isstruct(m_MF) ~= 1 && nargin > 5 % SV of Taylor expansion
    epoch = varargin{2};
    m_SV = varargin{3};
    maxdeg_SV = sqrt(length(m_SV(:,1))+1)-1;
    N_koeff_SV = maxdeg_SV*(maxdeg_SV+2);
    % extract quadratic variation
    if size(m_SV, 2) > 1
        index = find(m_SV(:,2) == 0);
        N_koeff_SV2 = index(1)-1;
        maxdeg_SV2 = sqrt(N_koeff_SV2+1)-1;
    else
        maxdeg_SV2 = 0;
    end
    % extract cubic variation
    if size(m_SV, 2) > 2
        index = find(m_SV(:,3) == 0);
        N_koeff_SV3 = index(1)-1;
        maxdeg_SV3 = sqrt(N_koeff_SV3+1)-1;
    else
        maxdeg_SV3 = 0;
    end
end

B_MF = zeros(N_data, 3);
for i=1:N_Data_Frac:N_data
    index = [i:min(i+N_Data_Frac-1, N_data)]';
    [A_r, A_theta, A_phi] = design_SHA(r(index)/a, theta(index)*rad, phi(index)*rad, maxdeg_MF, 'int');
    
    if nargin == 4 % no SV
        B_MF(index, :) = [A_r*m_MF A_theta*m_MF A_phi*m_MF];        
    elseif isstruct(m_MF) == 1  % spline SV variation
%         g_i = zeros(m_MF.dim, length(index));
%         index_middle = find(m_MF.breaks(1) < t(index) & t(index) < m_MF.breaks(end));
%         g_i(:,index_middle) = ppval(t(index(index_middle))', m_MF);
        g_i = ppval(t(index)', m_MF);
        if nargin == 6
            boundary = varargin{2};
        else
            boundary = 'linear';
        end

        index_left = find(t(index) < m_MF.breaks(1));
        index_right = find(t(index) > m_MF.breaks(end));

        % ensure that scalar index_left=[] and index_right=[] become empty
        % matrices 0-by-1
        if length(index)==1 && isempty(index_left)
            index_left = ones(0, 1);
        end
        if length(index)==1 && isempty(index_right)
            index_right = ones(0, 1);
        end        

        if strcmpi(boundary(1), 'n')
            % spline extrapolation outside knots interval
%             g_i(:,index_left) = ppval(t(index(index_left))', m_MF);
%             g_i(:,index_right) = ppval(t(index(index_right))', m_MF);
        elseif strcmpi(boundary(1), 'c')
            % replace values outside time period with constant values
            g_left = ppval(m_MF.breaks(1),m_MF);
            g_right = ppval(m_MF.breaks(end),m_MF);
            g_i(:,index_left) = repmat(g_left, 1, length(index_left));
            g_i(:,index_right) = repmat(g_right, 1, length(index_right));
        elseif strcmpi(boundary(1), 'l')
            % replace values outside time period with linear extrapolation
            g_left = ppval(m_MF.breaks(1),m_MF);
            g_right = ppval(m_MF.breaks(end),m_MF);
            dg_left = ppval(m_MF.breaks(1),fnderp(m_MF,1));
            dg_right = ppval(m_MF.breaks(end),fnderp(m_MF,1));
            g_i(:,index_left) = repmat(g_left, 1, length(index_left)) ...
                +  repmat(dg_left, 1, length(index_left)).*repmat((t(index(index_left))-m_MF.breaks(1)), 1, size(g_i,1))';
%                 +  repmat(dg_left, 1, length(index_left)).*repmat((t(index_left)-m_MF.breaks(1)), 1, size(g_i,1))';
            g_i(:,index_right) = repmat(g_right, 1, length(index_right)) ...
                +  repmat(dg_right, 1, length(index_right)).*repmat((t(index(index_right))-m_MF.breaks(end)), 1, size(g_i,1))';
%                 +  repmat(dg_right, 1, length(index_right)).*repmat((t(index_right)-m_MF.breaks(end)), 1, size(g_i,1))';
        else
            warning(['spline boundary ' boundary ' not defined. Linear extrapolation used.']);
        end
        B_MF(index, :) = [sum(A_r.*g_i',2) sum(A_theta.*g_i',2) sum(A_phi.*g_i',2)];
    else % Taylor expansion SV variation
        t_epoch= spdiags(2000 + t(index)/365.25 - epoch, 0, length(index), length(index));
        B_MF(index, :) = [A_r*m_MF A_theta*m_MF A_phi*m_MF] ...
            + [(t_epoch*A_r(:, 1:N_koeff_SV))*m_SV(:,1) ...
                (t_epoch*A_theta(:, 1:N_koeff_SV))*m_SV(:,1) ...
                (t_epoch*A_phi(:, 1:N_koeff_SV))*m_SV(:,1)];
        if maxdeg_SV2 > 0
            t_2 = spdiags((2000 + t(index)/365.25 - epoch).^2, 0, length(index), length(index));
            B_MF(index, :) = B_MF(index, :) ...
            + 1/2*[(t_2*A_r(:, 1:N_koeff_SV2))*m_SV(1:N_koeff_SV2,2) ...
                (t_2*A_theta(:, 1:N_koeff_SV2))*m_SV(1:N_koeff_SV2,2) ...
                (t_2*A_phi(:, 1:N_koeff_SV2))*m_SV(1:N_koeff_SV2,2)];
        end
        if maxdeg_SV3 > 0
            t_3 = spdiags((2000 + t(index)/365.25 - epoch).^3, 0, length(index), length(index));
            B_MF(index, :) = B_MF(index, :) ...
            + 1/6*[(t_3*A_r(:,     1:N_koeff_SV3))*m_SV(1:N_koeff_SV3,3) ...
                   (t_3*A_theta(:, 1:N_koeff_SV3))*m_SV(1:N_koeff_SV3,3) ...
                   (t_3*A_phi(:,   1:N_koeff_SV3))*m_SV(1:N_koeff_SV3,3)];
        end
    end
end

% external contributions
if nargin > 7
    Dst = varargin{4};
    if length(Dst)  == 1; Dst = Dst*ones(max_size); end;
    Dst = Dst(:);
    [A_r_ext, A_theta_ext, A_phi_ext] = design_SHA(r/a, theta*rad, phi*rad, maxdeg_ext, 'ext');
    [A_r, A_theta, A_phi] = design_SHA(r/a, theta*rad, phi*rad, maxdeg_ext, 'int');
    A_r_ext = [A_r_ext ...
            Dst.*(A_r_ext(:,1) + Q_1*A_r(:,1)) ...
            Dst.*(A_r_ext(:,2) + Q_1*A_r(:,2)) ...
            Dst.*(A_r_ext(:,3) + Q_1*A_r(:,3))];
    A_theta_ext = [A_theta_ext ... 
            Dst.*(A_theta_ext(:,1) + Q_1*A_theta(:,1)) ...
            Dst.*(A_theta_ext(:,2) + Q_1*A_theta(:,2)) ...
            Dst.*(A_theta_ext(:,3) + Q_1*A_theta(:,3))];
    A_phi_ext = [A_phi_ext ...
            Dst.*(A_phi_ext(:,1) + Q_1*A_phi(:,1)) ...
            Dst.*(A_phi_ext(:,2) + Q_1*A_phi(:,2)) ...
            Dst.*(A_phi_ext(:,3) + Q_1*A_phi(:,3))];
    
    m_ext = [m_ext_0(:,1); m_ext_Dst];
    if n_annual > 1
        for i = 1:length(m_ext_0(:,1))
            if n_annual == 3
                tmp = (1 + (m_ext_0(i,2)*cos(t/365.25*2*pi) + m_ext_0(i,3)*sin(t/365.25*2*pi))/m_ext_0(i,1));
            elseif n_annual == 5
                tmp = (1 + (m_ext_0(i,2)*cos(t/365.25*2*pi) + m_ext_0(i,3)*sin(t/365.25*2*pi)...
                    +  m_ext_0(i,4)*cos(t/365.25*4*pi) + m_ext_0(i,5)*sin(t/365.25*4*pi))/m_ext_0(i,1));
            else
                warning('n_annual should be 0,1,2');
            end
            A_r_ext(:,i)     = A_r_ext(:,i)    .* tmp;  
            A_theta_ext(:,i) = A_theta_ext(:,i).* tmp;
            A_phi_ext(:,i)   = A_phi_ext(:,i)  .* tmp;
        end
    end
    varargout{1} = [A_r_ext*m_ext A_theta_ext*m_ext A_phi_ext*m_ext];
end

function fprime = fnderp(f,dorder)
%FNDERP Differentiate a univariate function in ppform.
[breaks,coefs,l,k,d]=ppbrk(f);
knew=k-dorder;
for j=k-1:-1:knew
    coefs=coefs.*repmat([j:-1:j-k+1],d*l,1);
end
fprime=ppmak(breaks,coefs(:,1:knew),d);

function varargout = ppbrk(pp,varargin)
%PPBRK Part(s) of a ppform.
%
%   [BREAKS,COEFS,L,K,D] = PPBRK(PP)  breaks the ppform in PP into its parts 
%   and returns as many of them as are specified by the output arguments. 
%
%   PPBRK(PP)  returns nothing, but prints all parts.
%
%   OUT1 = PPBRK(PP,PART)  returns the particular part specified by the string 
%   PART, which may be (the beginning character(s) of) one of the following
%   strings: 'breaks', 'coefs', 'pieces' or 'l', 'order' or 'k',
%   'dim'ension, 'interval'.
%   For a while, there is also the choice
%      'guide'
%   that returns the coefficient array in the form required in the ppform used
%   in `A Practical Guide to Splines', especially for PPVALU there. This is not
%   available for vector-valued and/or tensor product splines.
%
%   PJ = PPBRK(PP,J)  returns the ppform of the J-th polynomial piece of the 
%   function in PP.
%
%   PC = PPBRK(PP,[A B])  returns the restriction/extension of the function
%   in PP to the interval  [A .. B], with [] causing PP to be returned as is.
%
%   If PP contains an m-variate spline and PART is not a string, then it
%   must be a cell-array, of length m .
%
%   [OUT1,...,OUTo] = PPBRK(PP, PART1,...,PARTi)  returns in OUTj the part 
%   specified by the string PARTj, j=1:o, provided o<=i.
%
%   Example: If PP contains a bivariate spline with at least 4 pieces
%   in the first variable, then
%
%      ppp = ppbrk(pp,{4,[-1 1]});
%
%   gives the bivariate spline that agrees with the given one on the
%   rectangle  [pp.breaks{1}(4) .. [pp.breaks{1}(5)] x [-1 1] .
%
%   See also SPBRK, FNBRK.

if ~isstruct(pp)
   if pp(1)~=10
      error('SPLINES:PPBRK:unknownfn',...
      'The input array does not seem to describe a function in ppform.')
   else
      ppi = pp;
      di=ppi(2); li=ppi(3); ki=ppi(5+li);

      pp = struct('breaks',reshape(ppi(3+(1:li+1)),1,li+1), ...
                  'coefs',reshape(ppi(5+li+(1:di*li*ki)),di*li,ki), ...
                  'form','pp', 'dim',di, 'pieces',li, 'order',ki);
   end
end 

if ~strcmp(pp.form,'pp')
   error('SPLINES:PPBRK:notpp',...
   'The input does not seem to describe a function in ppform.')
end
if nargin>1 % we have to hand back one or more parts
   np = max(1,nargout);
   if np>length(varargin)
      error('SPLINES:PPBRK:moreoutthanin', ...
            'Too many output arguments for the given input.')
   end
   varargout = cell(1,np);
   for jp=1:np
      part = varargin{jp};

      if ischar(part)
         if isempty(part)
	    error('SPLINES:PPBRK:partemptystr',...
	    'Part specification should not be an empty string.')
	 end
         switch part(1)
         case 'f',       out1 = [pp.form,'form'];
         case 'd',       out1 = pp.dim;
         case {'l','p'}, out1 = pp.pieces;
         case 'b',       out1 = pp.breaks;
         case 'o',       out1 = pp.order;
         case 'c',       out1 = pp.coefs;
	 case 'v',       out1 = length(pp.order);
         case 'g',       % if the spline is univariate, scalar-valued,
                         % return the coefs in the form needed in the ppform
                         % used in PGS.
            if length(pp.dim)>1||pp.dim>1||iscell(pp.order)
               error('SPLINES:PPBRK:onlyuniscalar', ...
                     ['''%s'' is only available for scalar-valued',...
                              ' univariate pp functions.'],part)
            else
               k = pp.order;
               out1 = (pp.coefs(:,k:-1:1).').* ...
	                repmat(cumprod([1 1:k-1].'),1,pp.pieces);
            end
         case 'i'
            if iscell(pp.breaks)
               for i=length(pp.order):-1:1
                  out1{i} = pp.breaks{i}([1 end]); end
            else
               out1 = pp.breaks([1 end]);
            end
         otherwise
            error('SPLINES:PPBRK:unknownpart',...
	    '''%s'' is not part of a ppform.',part)
         end
      elseif isempty(part)
	 out1 = pp;
      else % we are to restrict PP to some interval or piece
	 sizeval = pp.dim; if length(sizeval)>1, pp.dim = prod(sizeval); end
         if iscell(part)  % we are dealing with a tensor-product spline
   
            [breaks,c,l,k,d] = ppbrk(pp); m = length(breaks);
            sizec = [d,l.*k]; %size(c);
            if length(sizec)~=m+1
	       error('SPLINES:PPBRK:inconsistentfn', ...
	       'Information in PP is inconsistent.'),
            end
            for i=m:-1:1
               dd = prod(sizec(1:m));
               ppi = ppbrk1(ppmak(breaks{i},reshape(c,dd*l(i),k(i)),dd),...
                           part{i}) ;
               breaks{i} = ppi.breaks; sizec(m+1) = ppi.pieces*k(i);
               c = reshape(ppi.coefs,sizec);
               if m>1
                  c = permute(c,[1,m+1,2:m]);
                  sizec(2:m+1) = sizec([m+1,2:m]);
               end
            end
            out1 = ppmak(breaks,c, sizec);
   
         else  % we are dealing with a univariate spline
   
            out1 = ppbrk1(pp,part);
         end
         if length(sizeval)>1, out1 = fnchg(out1,'dz',sizeval); end
      end
      varargout{jp} = out1;
   end
else
   if nargout==0
     if iscell(pp.breaks) % we have a multivariate spline and, at present,
                          % I can't think of anything clever to do; so...
       disp(pp)
     else
       disp('breaks(1:l+1)'),        disp(pp.breaks)
       disp('coefficients(d*l,k)'),  disp(pp.coefs)
       disp('pieces number l'),      disp(pp.pieces)
       disp('order k'),              disp(pp.order)
       disp('dimension d of target'),disp(pp.dim)
       % disp('dimension v of domain'),disp(length(pp.order))
     end
   else
      varargout = {pp.breaks, pp.coefs, pp.pieces, pp.order, pp.dim};
   end
end

function pppart = ppbrk1(pp,part)
%PPBRK1 restriction of pp to some piece or interval

if isempty(part)||ischar(part), pppart = pp; return, end

if size(part,2) > 1 , % extract the part relevant to the interval 
                      % specified by  part =: [a b]  
   pppart = ppcut(pp,part(1,1:2));
else                  % extract the part(1)-th polynomial piece of pp (if any)
   pppart = pppce(pp,part(1));
end

function ppcut = ppcut(pp,interv)
%PPCUT returns the part of pp  specified by the interval interv =: [a b]  

xl = interv(1); xr = interv(2); if xl>xr, xl = xr; xr = interv(1);  end
if xl==xr
   warning('SPLINES:PPBRK:PPCUT:trivialinterval', ...
           'No changes made since the given end points are equal.')
   ppcut = pp; return
end
 
%  the first pol. piece is  jl ,
% the one responsible for argument  xl
jl=pp.pieces; index=find(pp.breaks(2:jl)>xl); 
                                   % note that the resulting  index  ...
if (~isempty(index)), jl=index(1); % ... is shifted down by one  ...
end                                % ... because of  breaks(2: ...
%  if xl ~= breaks(jl), recenter the pol.coeffs.
x=xl-pp.breaks(jl);
di = pp.dim;
if x ~= 0
   a=pp.coefs(di*jl+(1-di:0),:);
   for ii=pp.order:-1:2
      for i=2:ii
         a(:,i)=x*a(:,i-1)+a(:,i);
      end
   end
   pp.coefs(di*jl+(1-di:0),:)=a;
end
 
%  the last pol. piece is  jr ,
% the one responsible for argument  xr .
jr=pp.pieces;index=find(pp.breaks(2:jr+1)>=xr); 
                                   % note that the resulting ...
if (~isempty(index)), jr=index(1); % index  is shifted down by
end                                % ... one because of  breaks(2: ...
 
%  put together the cut-down  pp
di = pp.dim;
ppcut = ppmak([xl pp.breaks(jl+1:jr) xr], ...
                        pp.coefs(di*(jl-1)+(1:di*(jr-jl+1)),:),di);

function pppce = pppce(pp,j)
%PPPCE returns the j-th polynomial piece of pp  (if any).

%  if  pp  has a  j-th  piece, ...
if (0<j)&&(j<=pp.pieces)  %             ...  extract it
   di = pp.dim;
   pppce = ppmak([pp.breaks(j) pp.breaks(j+1)], ...
              pp.coefs(di*j+(1-di:0),:),di);
else
   error('SPLINES:PPBRK:wrongpieceno', ...
   'The given pp function does not have %g pieces.',j);
end

function pp = ppmak(breaks,coefs,d)
%PPMAK Put together a spline in ppform.
%
%   PPMAK(BREAKS,COEFS)  puts together a spline in ppform from the breaks
%   BREAKS and coefficient matrix COEFS. Each column of COEFS is
%   taken to be one coefficient, i.e., the spline is taken to be D-vector
%   valued if COEFS has D rows. Further, with L taken as length(BREAKS)-1,
%   the order K of the spline is computed as (# cols(COEFS))/L, and COEFS is
%   interpreted as a three-dimensional array of size [D,K,L], with
%   COEFS(i,:,j) containing the local polynomial coefficients for the i-th
%   component of the j-th polynomial piece, from highest to lowest.
%
%   PPMAK  will prompt you for BREAKS and COEFS.
%
%   PPMAK(BREAKS,COEFS,D), with D a positive integer, interprets the matrix
%   COEFS to be of size [D,L,K], with COEFS(i,j,:) containing the local
%   polynomial coefficients, from highest to lowest, of the i-th component
%   of the j-th polynomial piece.
%   In particular, the order K is taken to be the last dimension of COEFS,
%   and L is taken to be length(COEFS(:))/(D*K),
%   and BREAKS is expected to be of length L+1.
%   The toolbox uses internally only this second format, reshaping COEFS
%   to be of size [D*L,K].
%
%   For example,  ppmak([1 3 4],[1 2 5 6;3 4 7 8])  and
%                 ppmak([1 3 4],[1 2;3 4;5 6;7 8],2)
%   specify the same function (2-vector-valued, of order 2).
%
%   PPMAK(BREAKS,COEFS,SIZEC), with SIZEC a vector of positive integers,
%   interprets COEFS to be of size SIZEC =: [D,L,K], with COEFS(i,j,:)
%   containing the polynomial coefficient, from highest to lowest, of the i-th
%   component of the j-th polynomial piece. The dimension of the function's
%   target is taken to be SIZEC(1:end-2). Internally, COEFS is reshaped into a
%   matrix, of size [prod(SIZEC(1:end-1)),K].
%
%   For example, to make up the constant function, with basic interval [0..1]
%   say, whose value is the matrix EYE(2), you have to use the command
%      ppmak(0:1, eye(2), [2,2,1,1]);
%
%   PPMAK({BREAKS1,...,BREAKSm},COEFS)  puts together an m-variate
%   tensor-product spline in ppform. In this case, COEFS is expected to be of
%   size [D,lk], with lk := [l1*k1,...,lm*km] and li = length(BREAKS{i})-1,
%   all i, and this defines D and k := [k1,...,km].  If, instead, COEFS is
%   only an m-dimensional array, then D is taken to be 1.
%
%   PPMAK({BREAKS1,...,BREAKSm},COEFS,SIZEC)  uses the optional third argument
%   to specify the size of COEFS. The intended size of COEFS is needed in case
%   one or more of its trailing dimensions is a singleton and thus COEFS by
%   itself appears to be of lower dimension.
%
%   For example, if we intend to construct a 2-vector-valued bivariate
%   polynomial on the rectangle [-1 .. 1] x [0 .. 1], linear in the first
%   variable and constant in the second, say
%      coefs = zeros(2,2,1); coefs(:,:,1) = [1 0; 0 1];
%   then the straightforward
%      pp = ppmak({[-1 1],[0 1]},coefs);
%   will fail, producing a scalar-valued function of order 2 in each variable,
%   as will
%      pp = ppmak({[-1 1],[0 1]},coefs,size(coefs));
%   while the command
%      pp = ppmak({[-1 1],[0 1]},coefs,[2 2 1]);
%   will succeed.
%

if nargin==0
   breaks=input('Give the (l+1)-vector of breaks  >');
   coefs=input('Give the (d by (k*l)) matrix of local pol. coefficients  >');
end

sizec = size(coefs);

if iscell(breaks)  % we are dealing with a tensor-product spline
   if nargin>2
      if prod(sizec)~=prod(d)
        error('SPLINES:PPMAK:coefsdontmatchsize', ...
	      'The coefficient array is not of the explicitly specified size.')
      end, sizec = d;
   end
   m = length(breaks);
   if length(sizec)<m
      error('SPLINES:PPMAK:coefsdontmatchbreaks', ...
           ['If BREAKS is a cell-array of length m, then COEFS must ',...
             'have at least m dimensions.'])
   end
   if length(sizec)==m,  % coefficients of a scalar-valued function
      sizec = [1 sizec];
   end
   sizeval = sizec(1:end-m); sizec = [prod(sizeval), sizec(end-m+(1:m))];
   coefs = reshape(coefs, sizec);

   d = sizec(1);
   for i=m:-1:1
      l(i) = length(breaks{i})-1;
      k(i) = fix(sizec(i+1)/l(i));
      if k(i)<=0||k(i)*l(i)~=sizec(i+1)
         error('SPLINES:PPMAK:piecesdontmatchcoefs', ...
	       ['The specified number %g of polynomial pieces is', ...
                ' incompatible\nwith the total number %g of coefficients', ...
                ' supplied in variable %g.'], l(i),sizec(i+1),i)
      end
      breaks{i} = reshape(breaks{i},1,l(i)+1);
   end
else
  if nargin<3
     if isempty(coefs)
        error('SPLINES:PPMAK:emptycoefs','The coefficient sequence is empty.')
     end
     sizeval = sizec(1:end-1);
     d = prod(sizeval); kl = sizec(end);
     l=length(breaks)-1;k=fix(kl/l);
     if (k<=0)||(k*l~=kl)
       error('SPLINES:PPMAK:piecesdontmatchcoefs', ...
            ['The specified number %g of polynomial pieces is',...
             ' incompatible\nwith the total number %g of coefficients',...
             ' supplied.'],l,kl);
        elseif any(diff(breaks)<0)
        error('SPLINES:PPMAK:decreasingbreaks', ...
	      'The break sequence should be nondecreasing.')
     elseif breaks(1)==breaks(l+1)
        error('SPLINES:PPMAK:extremebreakssame', ...
	      'The extreme breaks should be different.')
     else
        % the ppformat expects coefs in array  (d*l) by k, while the standard
        % input supplies them in an array d by (k*l) . This requires the
        % following shuffling, from  D+d(-1+K + k(-1+L))=D-d +(K-k)d + dkL
        % to  D+d(-1+L + l(-1+K)=D-d +(L-l)d + dlK .
        % This used to be handled by the following:
        % c=coefs(:); temp = ([1-k:0].'*ones(1,l)+k*ones(k,1)*[1:l]).';
        % coefs=[1-d:0].'*ones(1,kl)+d*ones(d,1)*(temp(:).');
        % coefs(:)=c(coefs);
        % Thanks to multidimensional arrays, we can now simply say
        coefs = reshape(permute(reshape(coefs,[d,k,l]),[1,3,2]),d*l,k);
     end
  else % in the univariate case, a scalar D only specifies the dimension of
       % the target and COEFS must be a matrix (though that is not checked for);
       % but if D is a vector, then it is taken to be the intended size of
       % COEFS whatever the actual dimensions of COEFS might be.
     if length(d)==1
        k = sizec(end); l = prod(sizec(1:end-1))/d;
     else
	if prod(d)~=prod(sizec)
	   error('SPLINES:PPMAK:coefsdontmatchsize', ...
	        ['The size of COEFS, [',num2str(sizec), ...
	         '], does not match the specified size, [',num2str(d),'].'])
        end
        k = d(end); l = d(end-1); d(end-1:end) = [];
	if isempty(d), d = 1; end
     end
     if l+1~=length(breaks)
        error('SPLINES:PPMAK:coefsdontmatchbreaks', ...
	      'COEFS indicates %g piece(s) while BREAKS indicates %g.', ...
	l, length(breaks)-1), end
     sizeval = d;
  end
  breaks = reshape(breaks,1,l+1);
end
pp.form = 'pp';
pp.breaks = breaks;
pp.coefs = coefs;
pp.pieces = l;
pp.order = k;
pp.dim = sizeval;

