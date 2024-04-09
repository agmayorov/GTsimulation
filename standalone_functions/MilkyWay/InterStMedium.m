function  [Den, ChemComp] = InterStMedium(X, Y, Z)
%   clear; X=0; Y=0.1:0.1:16; Z=0.1; % Just for tests
%   Arguments:
%       X, Y, Z     - Float         -   Position, kpc
%   Output:  
%       Den         - Float         -   Total mass density, in g/cm3
%       ChemComp    - Float arr.    -   Fraction of (He, O, N2, O2, Ar, H, N) (O+ N+ H+ He+ O2+ NO+)

    %% Conversion to CYL RF
    [~, R, z] = cart2pol(X, Y, Z);
    if isrow(R), R = R'; end

    %% Initialization
    [nTot, nHII, nHI, nH2] = deal(zeros(length(R),1)); %#ok<ASGLU>

    %% Ionized hydrohen
    n = [0.025 0.200];  % kpc
    h = [1 0.15];
    Ri = [0 4];         % kpc
    a = [20 2];         % kpc

    nHII = zeros(length(R),1);
    for i = 1:2
        nHII = nHII + n(i).*exp(-abs(z)./h(i)-(R-Ri(i)).^2./a(i)^2); %sm^(-3)
    end

    %% Atomic hydrogen
    nGB = 0.33; %sm^(-3) Concentration of the HI atoms in the disk at location 4 <R <8 kpc in model (Gordon & Burton, 1976)
    nDL = 0.57; %sm^(-3) Concentration of the HI atoms in the disk at location 4 <R <8 kpc in model (Dickey & Lockman, 1990)
    A = [0.395 0.107 0.064];
    zi = [0.106 0.265 0.403 0.0523];

    persistent Us e0s z0s zhs
    if isempty(Us)
        load('S.mat');  %n(HI) at b = 0�, sm^(-3) (Gordon & Burton, 1976)
        %%U=nHII; % Spline
    end

    nHIa=zeros(length(R),1);
    for i = 1:2
        nHIa = nHIa + A(i).*exp(-log(2)*z.^2./zi(i).^2);
    end;nHIa = nHIa + A(3)*exp(-abs(z)./zi(3));    %Distribution from (Dickey & Lockman, 1990)
    nHIb = nDL*exp(-z.^2.*exp(-0.22*R)./zi(4)^2);   %Distribution from (Cox et al., 1986)
    
    nHI = (Us(R)/nGB).*( nHIa.*(R<=9) + nHIb.*(R>9) ); % 8 - 10 interpolation
    
    %% Molecular hydrogen
    Xco = 1.9*10^20; % Xco = NH2/Wco, molecules/(sm^2 K km) sec - conversion factor (Strong & Mattox, 1996)

    nH2 = 3.24*10^(-22)*Xco*e0s(R).*exp((-log(2)*(z-z0s(R)).^2)./zhs(R).^2);    %sm^(-3)
    %%e0(R) K km/sec - CO volumetric luminosity
    %%z0(R), zh(R) - the characteristic scale of height and distribution width
    
    %% Total number of H atoms per volume
    nTot = nHI + 2*nH2 + nHII;  % molecules/sm^3
    %plot(R, nTot);
    Den = nTot*0.167e-23;

    ChemComp = [0 0 0 0 0 nHI 0 0 0 nHII 0 0 0 nH2];
    ChemComp = ChemComp./sum(ChemComp);
    