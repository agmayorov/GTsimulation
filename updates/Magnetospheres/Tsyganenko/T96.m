function [bx, by, bz] = T96(parmod, ps, x, y, z)
%   clear -global; [bx, by, bz] = T96(0, 15, 5, 5, [2 0 0 0])

    persistent a am0 s0 x00 dsig delimfx delimfy pdyn0 eps10
    if isempty(a)
        a = [1.162 22.344 18.50 2.602 6.903 5.287 0.5790 0.4462 0.7850];
        am0 = 70.;
        s0 = 1.08;
        x00 = 5.48;
        dsig = 0.005;
        delimfx = 20.;
        delimfy = 10.;    
        pdyn0 = 2.;
        eps10 = 3630.7;
    end

    pdyn  = parmod(1); % Solar wind pressure (nPa) / 1-10 quiet
    dst   = parmod(2); % Dst, nT / -10 - 10 quiet
    byimf = parmod(3); % Interplanetary magnetic filed, nT / -2 - 2 quiet
    bzimf = parmod(4);

    sps = sin(ps);
    depr= 0.8.*dst-13..*sqrt(pdyn);
    bt  = sqrt(byimf.^2+bzimf.^2);

    if byimf == 0. && bzimf == 0.
        theta=0.;
    else
        theta = atan2(byimf, bzimf);
        if theta <= 0
            theta = theta + 6.2831853;
        end  
    end

    ct  = cos(theta);
    st  = sin(theta);
    eps = 718.5.*sqrt(pdyn).*bt.*sin(theta./2.);

    facteps=eps./eps10-1.;
    factpd=sqrt(pdyn./pdyn0)-1.;

    rcampl = -a(1).*depr;
    tampl2 = a(2)+a(3).*factpd+a(4).*facteps;
    tampl3 = a(5)+a(6).*factpd;
    b1ampl = a(7)+a(8).*facteps;
    b2ampl = 20.*b1ampl;
    reconn = a(9);

    xappa = (pdyn./pdyn0).^0.14;
    xappa3= xappa.^3;
    ys = y.*ct - z.*st;
    zs = z.*ct + y.*st;

    factimf = exp(x./delimfx-(ys./delimfy).^2);

    oimfx = 0.;
    oimfy = reconn.*byimf.*factimf;
    oimfz = reconn.*bzimf.*factimf;

    rimfampl = reconn.*bt;

    xx = x.*xappa;
    yy = y.*xappa;
    zz = z.*xappa;

    x0   = x00./xappa;
    am   = am0./xappa;
    rho2 = y.^2+z.^2;
    asq  = am.^2;
    xmxm = am+x-x0;
    if xmxm < 0.
        xmxm = 0.;
    end
    axx0 = xmxm.^2;
    aro  = asq+rho2;
    sigma= sqrt((aro+axx0+sqrt((aro+axx0).^2-4.*asq.*axx0))./(2.*asq));

    if sigma < s0+dsig
        [cfx, cfy, cfz] = dipshld(ps, xx, yy, zz);
        [bxrc, byrc, bzrc, bxt2, byt2, bzt2, bxt3, byt3, bzt3] = tailrc96(sps, xx, yy, zz); 
        [r1x, r1y, r1z] = birk1tot_02(ps, xx, yy, zz);
        [r2x, r2y, r2z] = birk2tot_02(ps, xx, yy, zz);
        [rimfx, rimfys, rimfzs] = intercon(xx, ys.*xappa, zs.*xappa);
        rimfy = rimfys.*ct + rimfzs.*st;
        rimfz = rimfzs.*ct - rimfys.*st;

        fx = cfx.*xappa3 + rcampl.*bxrc + tampl2.*bxt2 + tampl3.*bxt3+b1ampl.*r1x + b2ampl.*r2x + rimfampl.*rimfx;
        fy = cfy.*xappa3 + rcampl.*byrc + tampl2.*byt2 + tampl3.*byt3+b1ampl.*r1y + b2ampl.*r2y + rimfampl.*rimfy;
        fz = cfz.*xappa3 + rcampl.*bzrc + tampl2.*bzt2 + tampl3.*bzt3+b1ampl.*r1z + b2ampl.*r2z + rimfampl.*rimfz;

        if sigma < s0-dsig
            bx = fx;
            by = fy;
            bz = fz;
        else
            fint = 0.5.*(1.-(sigma-s0)./dsig);
            fext = 0.5.*(1.+(sigma-s0)./dsig);

            [qx, qy, qz] = dipole(ps, x, y, z);
            
            bx = (fx+qx).*fint + oimfx.*fext - qx;
            by = (fy+qy).*fint + oimfy.*fext - qy;
            bz = (fz+qz).*fint + oimfz.*fext - qz;
        end
    else
        [qx, qy, qz] = dipole(ps, x, y, z);
        
        bx = oimfx - qx;
        by = oimfy - qy;
        bz = oimfz - qz;
    end
end


function [bx,by,bz] = dipshld(ps,x,y,z)
    persistent a1 a2
    if isempty(a1)
        a1 = [.24777 -27.003 -.46815 7.0637 -1.5918 -.90317E-01 57.522 13.757 2.0100 10.458 4.5798 2.1695];
        a2 = [-.65385 -18.061 -.40457 -5.0995 1.2846 .78231E-01 39.592 13.291 1.9970 10.062 4.5140,2.1558];
    end
    
    cps = cos(ps);
    sps = sin(ps);
    [hx, hy, hz] = cylharm(a1, x, y, z);
    [fx, fy, fz] = cylhar1(a2, x, y, z);

    bx = hx.*cps+fx.*sps;
    by = hy.*cps+fy.*sps;
    bz = hz.*cps+fz.*sps;
    
    return
end

function [bx, by, bz] = cylharm(a, x, y, z)

    rho = sqrt(y.^2 + z.^2);
    if rho < 1.0d-8
        sinfi = 1.0d0;
        cosfi = 0.0d0;
        rho = 1.0d-8;
    else
        sinfi = z./rho;
        cosfi = y./rho;
    end

    sinfi2 = sinfi.^2;
    si2co2 = sinfi2 - cosfi.^2;

    bx = 0.;
    by = 0.;
    bz = 0.;
    
    for i = 1:3
        dzeta= rho./a(i+6);
        xj0  = bes(dzeta,0);
        xj1  = bes(dzeta,1);
        xexp = exp(x./a(i+6));
        bx = bx - a(i).*xj1.*xexp.*sinfi;
        by = by + a(i).*(2.0d0.*xj1./dzeta-xj0).*xexp.*sinfi.*cosfi;
        bz = bz + a(i).*(xj1./dzeta.*si2co2-xj0.*sinfi2).*xexp;
    end

    for i = 4:6
        dzeta= rho./a(i+6);
        xksi = x./a(i+6);
        xj0  = bes(dzeta,0);
        xj1  = bes(dzeta,1);
        xexp = exp(xksi);
        brho = (xksi.*xj0-(dzeta.^2+xksi-1.0d0).*xj1./dzeta).*xexp.*sinfi;
        bphi = (xj0+xj1./dzeta.*(xksi-1.0d0)).*xexp.*cosfi;
        bx = bx + a(i).*(dzeta.*xj0+xksi.*xj1).*xexp.*sinfi;
        by = by + a(i).*(brho.*cosfi-bphi.*sinfi);
        bz = bz + a(i).*(brho.*sinfi+bphi.*cosfi);
    end

    return
end	

function [bx, by, bz] = cylhar1(a, x, y, z)

    rho = sqrt(y.^2 + z.^2);

    if rho < 1.D-10
        sinfi = 1.D0;
        cosfi = 0.D0;
    else
        sinfi = z./rho;
        cosfi = y./rho;
    end

    bx = 0.;
    by = 0.;
    bz = 0.;

    for i = 1:3
        dzeta= rho./a(i+6);
        xksi = x./a(i+6);
        xj0  = bes(dzeta,0);
        xj1  = bes(dzeta,1);
        xexp = exp(xksi);
        brho = xj1.*xexp;
        bx = bx - a(i).*xj0.*xexp;
        by = by + a(i).*brho.*cosfi;
        bz = bz + a(i).*brho.*sinfi;
    end

    for i = 4:6
        dzeta= rho./a(i+6);
        xksi = x./a(i+6);
        xj0  = bes(dzeta,0);
        xj1  = bes(dzeta,1);
        xexp = exp(xksi);
        brho = (dzeta.*xj0+xksi.*xj1).*xexp;
        bx = bx + a(i).*(dzeta.*xj1-xj0.*(xksi+1.0d0)).*xexp;
        by = by + a(i).*brho.*cosfi;
        bz = bz + a(i).*brho.*sinfi;
    end
end

function bes = bes(x, k)

    if k==0
        bes = bes0(x);
        return
    end

    if k==1 
        bes = bes1(x);
        return
    end

    if x==0
        bes = 0.0d0;
        return
    end

    g=2./x;
    
    if x > k
        n = 1;
        xjn = bes1(x);
        xjnm1 = bes0(x);

        while true
            xjnp1 = g.*n.*xjn - xjnm1;
            n = n + 1;
            if n < k 
                xjnm1 = xjn;
                xjn = xjnp1;
            else        
                bes = xjnp1;
                return
            end
        end
    end
    
    n = 24;
    xjn = 1.0;
    xjnp1 = 0.;
    sum = 0.;

    while true
        if rem(n,2)==0
            sum = sum + xjn;
        end
        xjnm1 = g.*n.*xjn - xjnp1;
        n = n-1;

        xjnp1 = xjn;
        xjn = xjnm1;
        if n==k
            bes = xjn;
        end

        if abs(xjn) > 1.0d5
            xjnp1 = xjnp1.*1.0d-5;
            xjn = xjn.*1.0d-5;
            sum = sum.*1.0d-5;
            if n<=k
                bes = bes.*1.0d-5;
            end
        end

        if n == 0
            sum = xjn + 2.0d0.*sum;
            bes = bes./sum;
            return
        end
    end

end

function bes0 = bes0(x)
    if abs(x) < 3
        x32 = (x./3.).^2;
        bes0 = 1.0d0-x32.*(2.2499997d0-x32.*(1.2656208d0-x32.*(0.3163866d0-x32.*(0.0444479d0-x32.*(0.0039444d0-x32.*0.00021d0)))));
    else
        xd3 = 3./x;
        f0 = 0.79788456d0-xd3.*(0.00000077d0+xd3.*(0.00552740d0+xd3.*(0.00009512d0-xd3.*(0.00137237d0-xd3.*(0.00072805d0-xd3.*0.00014476d0)))));
        t0 = x-0.78539816d0-xd3.*(0.04166397d0+xd3.*(0.00003954d0-xd3.*(0.00262573d0-xd3.*(0.00054125d0+xd3.*(0.00029333d0-xd3.*0.00013558d0)))));
        bes0 = f0./sqrt(x).*cos(t0);
    end
    
    return
end

function bes1 = bes1(x)
    if abs(x) < 3
        x32 = (x./3.0d0).^2;
        bes1xm1 = 0.5d0-x32.*(0.56249985d0-x32.*(0.21093573d0-x32.*(0.03954289d0-x32.*(0.00443319d0-x32.*(0.00031761d0-x32.*0.00001109d0)))));
        bes1 = bes1xm1.*x;
    else
        xd3 = 3.0d0./x;
        f1 = 0.79788456d0+xd3.*(0.00000156d0+xd3.*(0.01659667d0+xd3.*(0.00017105d0-xd3.*(0.00249511d0-xd3.*(0.00113653d0-xd3.*0.00020033d0)))));
        t1 = x-2.35619449d0+xd3.*(0.12499612d0+xd3.*(0.0000565d0-xd3.*(0.00637879d0-xd3.*(0.00074348d0+xd3.*(0.00079824d0-xd3.*0.00029166d0)))));
        bes1 = f1./sqrt(x).*cos(t1);
    end

    return
end

function [bx, by, bz] = intercon(x, y, z)
    persistent a
    if isempty(a)
        a = [-8.411078731 5932254.951 -9073284.93 -11.68794634 ...
             6027598.824 -9218378.368 -6.508798398 -11824.42793 18015.66212 ...
             7.99754043 13.9669886 90.24475036 16.75728834 1015.645781 ...
             1553.493216];
    end

    p(1:3) = a(10:12);
    r(1:3) = a(13:15);

    rp=zeros(1,3);
    rr=zeros(1,3);
    for i = 1:3
        rp(i)=1./p(i);
        rr(i)=1./r(i);
    end

    l = 0;
    bx = 0.;
    by = 0.;
    bz = 0.;

    for i = 1:3
        cypi = cos(y.*rp(i));
        sypi = sin(y.*rp(i));

        for k = 1:3
            szrk = sin(z.*rr(k));
            czrk = cos(z.*rr(k));
            sqpr = sqrt(rp(i).^2+rr(k).^2);
            epr  = exp(x.*sqpr);

            hx = -sqpr.*epr.*cypi.*szrk;
            hy = rp(i).*epr.*sypi.*szrk;
            hz = -rr(k).*epr.*cypi.*czrk;

            l=l+1;

            bx = bx + a(l).*hx;
            by = by + a(l).*hy;
            bz = bz + a(l).*hz;
        end
    end

    return
end

function [bxrc, byrc, bzrc, bxt2, byt2, bzt2, bxt3, byt3, bzt3] = tailrc96(sps, x, y, z)
    global cpss spss dpsrr rps warp d xs zs dxsx dxsy dxsz dzsx dzsy dzsz dzetas ddzetadx ddzetady ddzetadz zsww

    persistent arc atail2 atail3
    if isempty(arc)
        arc = [-3.087699646,3.516259114,18.81380577,-13.95772338, ...
           -5.497076303,0.1712890838,2.392629189,-2.728020808,-14.79349936, ...
           11.08738083,4.388174084,0.2492163197E-01,0.7030375685, ...
          -.7966023165,-3.835041334,2.642228681,-0.2405352424,-0.7297705678, ...
          -0.3680255045,0.1333685557,2.795140897,-1.078379954,0.8014028630, ...
          0.1245825565,0.6149982835,-0.2207267314,-4.424578723,1.730471572, ...
          -1.716313926,-0.2306302941,-0.2450342688,0.8617173961E-01, ...
           1.54697858,-0.6569391113,-0.6537525353,0.2079417515,12.75434981, ...
           11.37659788,636.4346279,1.752483754,3.604231143,12.83078674, ...
          7.412066636,9.434625736,676.7557193,1.701162737,3.580307144, ...
           14.64298662];

        atail2 = [.8747515218,-.9116821411,2.209365387,-2.159059518, ...
          -7.059828867,5.924671028,-1.916935691,1.996707344,-3.877101873, ...
          3.947666061,11.38715899,-8.343210833,1.194109867,-1.244316975, ...
          3.73895491,-4.406522465,-20.66884863,3.020952989,.2189908481, ...
          -.09942543549,-.927225562,.1555224669,.6994137909,-.08111721003, ...
          -.7565493881,.4686588792,4.266058082,-.3717470262,-3.920787807, ...
          .02298569870,.7039506341,-.5498352719,-6.675140817,.8279283559, ...
          -2.234773608,-1.622656137,5.187666221,6.802472048,39.13543412, ...
           2.784722096,6.979576616,25.71716760,4.495005873,8.068408272, ...
          93.47887103,4.158030104,9.313492566,57.18240483];

        atail3 = [-19091.95061,-3011.613928,20582.16203,4242.918430, ...
          -2377.091102,-1504.820043,19884.04650,2725.150544,-21389.04845, ...
          -3990.475093,2401.610097,1548.171792,-946.5493963,490.1528941, ...
          986.9156625,-489.3265930,-67.99278499,8.711175710,-45.15734260, ...
          -10.76106500,210.7927312,11.41764141,-178.0262808,.7558830028, ...
           339.3806753,9.904695974,69.50583193,-118.0271581,22.85935896, ...
          45.91014857,-425.6607164,15.47250738,118.2988915,65.58594397, ...
          -201.4478068,-14.57062940,19.69877970,20.30095680,86.45407420, ...
          22.50403727,23.41617329,48.48140573,24.61031329,123.5395974, ...
          223.5367692,39.50824342,65.83385762,266.2948657];
    end

    rh = 9.;
    dr = 4.;
    g  = 10.;
    d0 = 2.;
    deltady = 10.;

    dr2 = dr.*dr;
    c11 = sqrt((1.0d0+rh).^2+dr2);
    c12 = sqrt((1.0d0-rh).^2+dr2);
    c1  = c11-c12;
    spsc1 = sps./c1;
    rps = 0.5.*(c11+c12).*sps;

    r = sqrt(x.*x+y.*y+z.*z);
    sq1 = sqrt((r+rh).^2+dr2);
    sq2 = sqrt((r-rh).^2+dr2);
    c = sq1-sq2;
    cs = (r+rh)./sq1 - (r-rh)./sq2;
    spss  = spsc1./r.*c;
    cpss  = sqrt(1.0d0-spss.^2);
    dpsrr = sps./(r.*r).*(cs.*r-c)./sqrt((r.*c1).^2-(c.*sps).^2);

    wfac = y./(y.^4+1.0d4);
    w    = wfac.*y.^3;
    ws   = 4.0d4.*y.*wfac.^2;
    warp = g.*sps.*w;
    xs   = x.*cpss-z.*spss;
    zsww = z.*cpss+x.*spss;
    zs   = zsww +warp;

    dxsx = cpss-x.*zsww.*dpsrr;
    dxsy = -y.*zsww.*dpsrr;
    dxsz = -spss-z.*zsww.*dpsrr;
    dzsx = spss+x.*xs.*dpsrr;
    dzsy = xs.*y.*dpsrr  +g.*sps.*ws;
    dzsz = cpss+xs.*z.*dpsrr;

    d = d0+deltady.*(y./20.0d0).^2;
    dddy = deltady.*y.*0.005d0;

    dzetas   = sqrt(zs.^2+d.^2);
    ddzetadx = zs.*dzsx./dzetas;
    ddzetady = (zs.*dzsy+d.*dddy)./dzetas;
    ddzetadz = zs.*dzsz./dzetas;

    [wx, wy, wz] = shlcar3x3(arc, x, y, z, sps);
    [hx, hy, hz] = ringcurr96(x, y, z);
    bxrc = wx + hx;
    byrc = wy + hy;
    bzrc = wz + hz;

    [wx, wy, wz] = shlcar3x3(atail2, x, y, z, sps);
    [hx, hy, hz] = taildisk(x, y, z);
    bxt2 = wx + hx;
    byt2 = wy + hy;
    bzt2 = wz + hz;

    [wx, wy, wz] = shlcar3x3(atail3, x, y, z, sps);
    [hx, hz] = tail87(x, z);
    bxt3 = wx + hx;
    byt3 = wy;
    bzt3 = wz + hz;

    return
end

function [bx, by, bz] = ringcurr96(x, y, z)
    global cpss spss dpsrr xs zs dxsx dxsy dxsz dzsx dzsy dzsz

    f = [569.895366D0 -1603.386993D0]; 
    beta = [2.722188 3.766875]; 

    d0 = 2.;
    deltadx = 0.;
    xd = 0.;
    xldx = 4.; 

    dzsy = xs.*y.*dpsrr;
    
    xxd  = x-xd;
    fdx  = 0.5d0.*(1.0d0+xxd./sqrt(xxd.^2+xldx.^2));
    dddx = deltadx.*0.5d0.*xldx.^2./sqrt(xxd.^2+xldx.^2).^3;
    d = d0 + deltadx.*fdx;

    dzetas = sqrt(zs.^2 + d.^2);
    
    rhos     = sqrt(xs.^2 + y.^2);
    ddzetadx = (zs.*dzsx + d.*dddx)./dzetas;
    ddzetady = zs.*dzsy./dzetas;
    ddzetadz = zs.*dzsz./dzetas;
    
    if rhos < 1.0d-5
        drhosdx = 0.;
        drhosdy = (abs(1.0d0).*sign(y+eps));
        drhosdz = 0.;
    else
        drhosdx = xs.*dxsx./rhos;
        drhosdy = (xs.*dxsy+y)./rhos;
        drhosdz = xs.*dxsz./rhos;
    end

    bx = 0.;
    by = 0.;
    bz = 0.;

    for i = 1:2
        bi = beta(i);

        s1 = sqrt((dzetas + bi).^2 + (rhos + bi).^2);
        s2 = sqrt((dzetas + bi).^2 + (rhos - bi).^2);
        ds1ddz = (dzetas + bi)./s1;
        ds2ddz = (dzetas + bi)./s2;
        ds1drhos = (rhos + bi)./s1;
        ds2drhos = (rhos - bi)./s2;

        ds1dx = ds1ddz.*ddzetadx + ds1drhos.*drhosdx;
        ds1dy = ds1ddz.*ddzetady + ds1drhos.*drhosdy;
        ds1dz = ds1ddz.*ddzetadz + ds1drhos.*drhosdz;

        ds2dx = ds2ddz.*ddzetadx + ds2drhos.*drhosdx;
        ds2dy = ds2ddz.*ddzetady + ds2drhos.*drhosdy;
        ds2dz = ds2ddz.*ddzetadz + ds2drhos.*drhosdz;

        s1ts2 = s1.*s2;
        s1ps2 = s1+s2;
        s1ps2sq = s1ps2.^2;
        fac1 = sqrt(s1ps2sq-(2.0d0.*bi).^2);
        as = fac1./(s1ts2.*s1ps2sq);
        term1 = 1.0d0./(s1ts2.*s1ps2.*fac1);
        fac2 = as./s1ps2sq;
        dasds1 = term1 - fac2./s1.*(s2.*s2+s1.*(3.0d0.*s1+4.0d0.*s2));
        dasds2 = term1 - fac2./s2.*(s1.*s1+s2.*(3.0d0.*s2+4.0d0.*s1));

        dasdx = dasds1.*ds1dx + dasds2.*ds2dx;
        dasdy = dasds1.*ds1dy + dasds2.*ds2dy;
        dasdz = dasds1.*ds1dz + dasds2.*ds2dz;

        bx = bx + f(i).*((2.0d0.*as+y.*dasdy).*spss-xs.*dasdz+as.*dpsrr.*(y.^2.*cpss+z.*zs));
        by = by - f(i).*y.*(as.*dpsrr.*xs+dasdz.*cpss+dasdx.*spss);
        bz = bz + f(i).*((2.0d0.*as+y.*dasdy).*cpss+xs.*dasdx-as.*dpsrr.*(x.*zs+y.^2.*spss));
    end
end

function [bx, by, bz] = taildisk(x, y, z)
    global xs dxsx dxsy dxsz dzetas ddzetadx ddzetady ddzetadz spss dpsrr cpss zsww
    
    xshift = 4.5;
    f = [-745796.7338 1176470.141 -444610.529 -57508.01028];
    beta = [7.9250000 8.0850000 8.4712500 27.89500];

    rhos = sqrt((xs-xshift).^2+y.^2);
    if rhos < 1.0d-5
        drhosdx = 0.0d0;
        drhosdy = (abs(1.0d0).*sign(y+eps));
        drhosdz = 0.0d0;
    else
        drhosdx = (xs-xshift).*dxsx./rhos;
        drhosdy = ((xs-xshift).*dxsy+y)./rhos;
        drhosdz = (xs-xshift).*dxsz./rhos;
    end

    bx = 0.;
    by = 0.;
    bz = 0.;

    for i = 1:4
        bi = beta(i);

        s1 =sqrt((dzetas+bi).^2 + (rhos+bi).^2);
        s2 =sqrt((dzetas+bi).^2 + (rhos-bi).^2);
        ds1ddz = (dzetas+bi)./s1;
        ds2ddz = (dzetas+bi)./s2;
        ds1drhos = (rhos+bi)./s1;
        ds2drhos = (rhos-bi)./s2;

        ds1dx = ds1ddz.*ddzetadx + ds1drhos.*drhosdx;
        ds1dy = ds1ddz.*ddzetady + ds1drhos.*drhosdy;
        ds1dz = ds1ddz.*ddzetadz + ds1drhos.*drhosdz;

        ds2dx = ds2ddz.*ddzetadx + ds2drhos.*drhosdx;
        ds2dy = ds2ddz.*ddzetady + ds2drhos.*drhosdy;
        ds2dz = ds2ddz.*ddzetadz + ds2drhos.*drhosdz;

        s1ts2 = s1.*s2;
        s1ps2 = s1+s2;
        s1ps2sq = s1ps2.^2;
        fac1 = sqrt(s1ps2sq-(2.0d0.*bi).^2);
        as = fac1./(s1ts2.*s1ps2sq);
        term1 = 1.0d0./(s1ts2.*s1ps2.*fac1);
        fac2 = as./s1ps2sq;
        dasds1 = term1-fac2./s1.*(s2.*s2+s1.*(3.0d0.*s1+4.0d0.*s2));
        dasds2 = term1-fac2./s2.*(s1.*s1+s2.*(3.0d0.*s2+4.0d0.*s1));

        dasdx = dasds1.*ds1dx + dasds2.*ds2dx;
        dasdy = dasds1.*ds1dy + dasds2.*ds2dy;
        dasdz = dasds1.*ds1dz + dasds2.*ds2dz;

        bx = bx + f(i).*((2.0d0.*as+y.*dasdy).*spss-(xs-xshift).*dasdz+as.*dpsrr.*(y.^2.*cpss+z.*zsww));
        by = by - f(i).*y.*(as.*dpsrr.*xs+dasdz.*cpss+dasdx.*spss);
        bz = bz + f(i).*((2.0d0.*as+y.*dasdy).*cpss+(xs-xshift).*dasdx-as.*dpsrr.*(x.*zsww+y.^2.*spss));
    end
end

function [bx, bz] = tail87(x, z)
    global rps warp
    persistent dd hpi rt xn x1 x2 b0 b1 b2 xn21 xnr adln
    if isempty(dd)
        dd=3.;
        hpi=1.5707963;
        rt=40.;
        xn=-10.;
        x1=-1.261;
        x2=-0.663;
        b0=0.391734;
        b1=5.89715;
        b2=24.6833;
        xn21=76.37;
        xnr=-0.1071;
        adln=0.13238005;
    end
    
    zs = z - rps + warp;
    zp = z - rt;
    zm = z + rt;

    xnx = xn - x;
    xnx2= xnx.^2;
    xc1 = x - x1;
    xc2 = x - x2;
    xc22= xc2.^2;
    xr2 = xc2.*xnr;
    xc12= xc1.^2;
    d2  = dd.^2;
    
    b20 = zs.^2 + d2;
    b2p = zp.^2 + d2;
    b2m = zm.^2 + d2;
    b  = sqrt(b20);
    bp = sqrt(b2p);
    bm = sqrt(b2m);
    
    xa1  = xc12 + b20;
    xap1 = xc12 + b2p;
    xam1 = xc12 + b2m;
    xa2  = 1./(xc22+b20);
    xap2 = 1./(xc22+b2p);
    xam2 = 1./(xc22+b2m);
    xna  = xnx2 + b20;
    xnap = xnx2 + b2p;
    xnam = xnx2 + b2m;
    
    f  = b20 - xc22;
    fp = b2p - xc22;
    fm = b2m - xc22;
    
    xln1  = log(xn21./xna);
    xlnp1 = log(xn21./xnap);
    xlnm1 = log(xn21./xnam);
    xln2  = xln1  + adln;
    xlnp2 = xlnp1 + adln;
    xlnm2 = xlnm1 + adln;
    aln   = 0.25.*(xlnp1+xlnm1-2..*xln1);
    
    s0  = (atan(xnx./b)+hpi)./b;
    s0p = (atan(xnx./bp)+hpi)./bp;
    s0m = (atan(xnx./bm)+hpi)./bm;
    s1  = (xln1.*.5+xc1.*s0)./xa1;
    s1p = (xlnp1.*.5+xc1.*s0p)./xap1;
    s1m = (xlnm1.*.5+xc1.*s0m)./xam1;
    s2  = (xc2.*xa2.*xln2-xnr-f.*xa2.*s0).*xa2;
    s2p = (xc2.*xap2.*xlnp2-xnr-fp.*xap2.*s0p).*xap2;
    s2m = (xc2.*xam2.*xlnm2-xnr-fm.*xam2.*s0m).*xam2;
    
    g1  = (b20.*s0-0.5.*xc1.*xln1)./xa1;
    g1p = (b2p.*s0p-0.5.*xc1.*xlnp1)./xap1;
    g1m = (b2m.*s0m-0.5.*xc1.*xlnm1)./xam1;
    g2  = ((0.5.*f.*xln2+2..*s0.*b20.*xc2).*xa2+xr2).*xa2;
    g2p = ((0.5.*fp.*xlnp2+2..*s0p.*b2p.*xc2).*xap2+xr2).*xap2;
    g2m = ((0.5.*fm.*xlnm2+2..*s0m.*b2m.*xc2).*xam2+xr2).*xam2;
    
    bx = b0.*(zs.*s0-0.5.*(zp.*s0p+zm.*s0m))+b1.*(zs.*s1-0.5.*(zp.*s1p+zm.*s1m))+b2.*(zs.*s2-0.5.*(zp.*s2p+zm.*s2m));
    bz = b0.*aln+b1.*(g1-0.5.*(g1p+g1m))+b2.*(g2-0.5.*(g2p+g2m));
end

function [hx, hy, hz] = shlcar3x3(a, x, y, z, sps)
    cps = sqrt(1.0d0-sps.^2);
    s3ps= 4.0d0.*cps.^2-1.0d0;
    
    l = 0;
    hx = 0.;
    hy = 0.;
    hz = 0.;
        
    for m = 1:2

        for i = 1:3
            p = a(36+i);
            q = a(42+i);
            cypi = cos(y./p);
            cyqi = cos(y./q);
            sypi = sin(y./p);
            syqi = sin(y./q);

            for k = 1:3
                r = a(39+k);
                s = a(45+k);
                szrk = sin(z./r);
                czsk = cos(z./s);
                czrk = cos(z./r);
                szsk = sin(z./s);
                sqpr = sqrt(1.0d0./p.^2+1.0d0./r.^2);
                sqqs = sqrt(1.0d0./q.^2+1.0d0./s.^2);
                epr  = exp(x.*sqpr);
                eqs  = exp(x.*sqqs);

                for n = 1:2
                    l = l+1;
                    if m == 1
                        if n == 1
                            dx = -sqpr.*epr.*cypi.*szrk;
                            dy = epr./p.*sypi.*szrk;
                            dz = -epr./r.*cypi.*czrk;
                            hx = hx + a(l).*dx;
                            hy = hy + a(l).*dy;
                            hz = hz + a(l).*dz;
                        else
                            dx = dx.*cps;
                            dy = dy.*cps;
                            dz = dz.*cps;
                            hx = hx + a(l).*dx;
                            hy = hy + a(l).*dy;
                            hz = hz + a(l).*dz;
                        end
                    else
                        if n == 1
                            dx = -sps.*sqqs.*eqs.*cyqi.*czsk;
                            dy = sps.*eqs./q.*syqi.*czsk;
                            dz = sps.*eqs./s.*cyqi.*szsk;
                            hx = hx+ a(l).*dx;
                            hy = hy + a(l).*dy;
                            hz = hz + a(l).*dz;
                        else
                            dx = dx.*s3ps;
                            dy = dy.*s3ps;
                            dz = dz.*s3ps;
                            hx = hx + a(l).*dx;
                            hy = hy + a(l).*dy;
                            hz = hz + a(l).*dz;
                        end
                    end
                end
            end
        end
    end
end

function [bx, by, bz] = birk1tot_02(ps, x, y, z)
    global dx scalein scaleout xx1 yy1 xx2 yy2 zz2 rh dr tilt xcentre radius dipx dipy
    persistent c1 c2
    if isempty(c1)
        c1 = [-0.911582E-03 -0.376654E-02 -0.727423E-02 -0.270084E-02 ...
          -0.123899E-02 -0.154387E-02 -0.340040E-02 -0.191858E-01 ...
          -0.518979E-01 0.635061E-01 0.440680 -0.396570 0.561238E-02 ...
           0.160938E-02 -0.451229E-02 -0.251810E-02 -0.151599E-02 ...
          -0.133665E-02 -0.962089E-03 -0.272085E-01 -0.524319E-01 ...
           0.717024E-01 0.523439 -0.405015 -89.5587 23.2806];

        c2 = [6.04133 .305415 .606066E-02 .128379E-03 -.179406E-04  ...
          1.41714 -27.2586 -4.28833 -1.30675 35.5607 8.95792 .961617E-03  ...
          -.801477E-03 -.782795E-03 -1.65242 -16.5242 -5.33798 .424878E-03  ...
          .331787E-03 -.704305E-03 .844342E-03 .953682E-04 .886271E-03  ...
          25.1120 20.9299 5.14569 -44.1670 -51.0672 -1.87725 20.2998  ...
          48.7505 -2.97415 3.35184 -54.2921 -.838712 -10.5123 70.7594  ...
          -4.94104 .106166E-03 .465791E-03 -.193719E-03 10.8439 -29.7968  ...
           8.08068 .463507E-03 -.224475E-04 .177035E-03 -.317581E-03  ...
          -.264487E-03 .102075E-03 7.71390 10.1915 -4.99797 -23.1114  ...
         -29.2043 12.2928 10.9542 33.6671 -9.3851 .174615E-03 -.789777E-06  ...
          .686047E-03 .460104E-04 -.345216E-02 .221871E-02 .110078E-01  ...
          -.661373E-02 .249201E-02 .343978E-01 -.193145E-05 .493963E-05  ...
          -.535748E-04 .191833E-04 -.100496E-03 -.210103E-03 -.232195E-02  ...
          .315335E-02 -.134320E-01 -.263222E-01];
    end

    tilt = 1.00891;
    xcentre = [2.28397 -5.60831];
    radius = [1.86106 7.83281];
    dipx = 1.12541;
    dipy = 0.945719;

    dx = -0.16D0;
    scalein = 0.08D0;
    scaleout = 0.4D0;
    xx1 = [-11.D0 -7.D0 -7.D0 -3.D0 -3.D0 1.D0 1.D0 1.D0 5.D0 5.D0 9.D0 9.D0];
    yy1 = [2.D0 0.D0 4.D0 2.D0 6.D0 0.D0 4.D0 8.D0 2.D0 6.D0 0.D0 4.D0];
    xx2 = [-10.D0 -7.D0 -4.D0 -4.D0 0.D0 4.D0 4.D0 7.D0 10.D0 0.D0 0.D0 0.D0 0.D0 0.D0];
    yy2 = [3.D0 6.D0 3.D0 9.D0 6.D0 3.D0 9.D0 6.D0 3.D0 0.D0 0.D0 0.D0 0.D0 0.D0];
    zz2 = [20.D0 20.D0 4.D0 20.D0 4.D0 4.D0 20.D0 20.D0 20.D0 2.D0 3.D0 4.5D0 7.D0 10.D0];

    rh = 9.D0; 
    dr = 4.D0;
    xltday = 78.D0;
    xltnght = 70.D0;
    dtet0 = 0.034906;

    tnoonn = (90.0d0-xltday).*0.01745329d0;
    tnoons = 3.141592654d0-tnoonn;

    dtetdn = (xltday-xltnght).*0.01745329d0;
    dr2 = dr.^2;

    sps = sin(ps);
    r2  = x.^2+y.^2+z.^2;
    r   = sqrt(r2);
    r3  = r.*r2;

    rmrh = r - rh;
    rprh = r + rh;
    sqm  = sqrt(rmrh.^2+dr2);
    sqp  = sqrt(rprh.^2+dr2);
    c = sqp - sqm;
    q = sqrt((rh+1.0d0).^2+dr2)-sqrt((rh-1.0d0).^2+dr2);
    spsas = sps./r.*c./q;
    cpsas = sqrt(1.0d0-spsas.^2);
    xas = x.*cpsas-z.*spsas;
    zas = x.*spsas+z.*cpsas;
    if xas ~= 0.0d0 || y ~= 0.0d0
        pas = atan2(y, xas);
    else
        pas = 0.0d0;
    end

    tas = atan2(sqrt(xas.^2+y.^2),zas);
    stas= sin(tas);
    f   = stas./(stas.^6.*(1.0d0-r3)+r3).^0.1666666667d0;

    tet0 = asin(f);
    if tas > 1.5707963d0
        tet0 = 3.141592654d0 - tet0;
    end
    dtet = dtetdn.*sin(pas.*0.5d0).^2;
    tetr1n = tnoonn + dtet;
    tetr1s = tnoons - dtet;

    if tet0 < tetr1n - dtet0 || tet0 > tetr1s + dtet0
        loc = 1;
    end
    if tet0 > tetr1n + dtet0 && tet0 < tetr1s - dtet0
        loc = 2;
    end
    if tet0 >= tetr1n - dtet0 && tet0 <= tetr1n + dtet0
        loc = 3;
    end
    if tet0 >= tetr1s - dtet0 && tet0 <= tetr1s + dtet0
        loc = 4;
    end

    xi = zeros(1,4);
    
    if loc == 1
        xi(1) = x;
        xi(2) = y;
        xi(3) = z;
        xi(4) = ps;
        d1 = diploop1(xi);
        bx = 0.0d0;
        by = 0.0d0;
        bz = 0.0d0;
        for i = 1:26
            bx = bx + c1(i).*d1(1,i);
            by = by + c1(i).*d1(2,i);
            bz = bz + c1(i).*d1(3,i);
        end
    end

    if loc == 2
        xi(1) = x;
        xi(2) = y;
        xi(3) = z;
        xi(4) = ps;
        d2 = condip1(xi);
        bx = 0.0d0;
        by = 0.0d0;
        bz = 0.0d0;
        for i = 1:79
            bx = bx + c2(i).*d2(1,i);
            by = by + c2(i).*d2(2,i);
            bz = bz + c2(i).*d2(3,i);
        end
    end

    if loc == 3
        t01 = tetr1n - dtet0;
        t02 = tetr1n + dtet0;
        sqr = sqrt(r);
        st01as = sqr./(r3+1.0d0./sin(t01).^6-1.0d0).^0.1666666667;
        st02as = sqr./(r3+1.0d0./sin(t02).^6-1.0d0).^0.1666666667;
        ct01as = sqrt(1.0d0-st01as.^2);
        ct02as = sqrt(1.0d0-st02as.^2);
        xas1 = r.*st01as.*cos(pas);
        y1   =  r.*st01as.*sin(pas);
        zas1 =r.*ct01as;
        x1   = xas1.*cpsas+zas1.*spsas;
        z1   = -xas1.*spsas+zas1.*cpsas;

        xi(1) = x1;
        xi(2) = y1;
        xi(3) = z1;
        xi(4) = ps;
        d1 = diploop1(xi);
        bx1 = 0.0d0;
        by1 = 0.0d0;
        bz1 = 0.0d0;
        for i = 1:26
            bx1 = bx1 + c1(i).*d1(1,i);
            by1 = by1 + c1(i).*d1(2,i);
            bz1 = bz1 + c1(i).*d1(3,i);
        end
        
        xas2= r.*st02as.*cos(pas);
        y2  =  r.*st02as.*sin(pas);
        zas2= r.*ct02as;
        x2  = xas2.*cpsas+zas2.*spsas;
        z2  = -xas2.*spsas+zas2.*cpsas;

        xi(1) = x2;
        xi(2) = y2;
        xi(3) = z2;
        xi(4) = ps;
        d2 = condip1(xi);
        bx2 = 0.0d0;
        by2 = 0.0d0;
        bz2 = 0.0d0;

        for i=1:79
            bx2 = bx2 + c2(i).*d2(1,i);
            by2 = by2 + c2(i).*d2(2,i);
            bz2 = bz2 + c2(i).*d2(3,i);
        end

        ss  = sqrt((x2-x1).^2+(y2-y1).^2+(z2-z1).^2);
        ds  = sqrt((x-x1).^2+(y-y1).^2+(z-z1).^2);
        frac=ds./ss;
        
        bx = bx1.*(1.0d0-frac)+bx2.*frac;
        by = by1.*(1.0d0-frac)+by2.*frac;
        bz = bz1.*(1.0d0-frac)+bz2.*frac;
    end

    if loc == 4
        t01 = tetr1s - dtet0;
        t02 = tetr1s + dtet0;
        sqr = sqrt(r);
        st01as = sqr./(r3+1.0d0./sin(t01).^6-1.0d0).^0.1666666667;
        st02as = sqr./(r3+1.0d0./sin(t02).^6-1.0d0).^0.1666666667;
        ct01as = -sqrt(1.0d0-st01as.^2);
        ct02as = -sqrt(1.0d0-st02as.^2);
        
        xas1= r.*st01as.*cos(pas);
        y1  =  r.*st01as.*sin(pas);
        zas1=r.*ct01as;
        x1  = xas1.*cpsas+zas1.*spsas;
        z1  = -xas1.*spsas+zas1.*cpsas;

        xi(1) = x1;
        xi(2) = y1;
        xi(3) = z1;
        xi(4) = ps;
        d2 = condip1(xi);
        bx1 = 0.0d0;
        by1 = 0.0d0;
        bz1 = 0.0d0;
        for i = 1:79
            bx1 = bx1 + c2(i).*d2(1,i);
            by1 = by1 + c2(i).*d2(2,i);
            bz1 = bz1 + c2(i).*d2(3,i);
        end
        xas2= r.*st02as.*cos(pas);
        y2  = r.*st02as.*sin(pas);
        zas2= r.*ct02as;
        x2  = xas2.*cpsas+zas2.*spsas;
        z2  = -xas2.*spsas+zas2.*cpsas;

        xi(1) = x2;
        xi(2) = y2;
        xi(3) = z2;
        xi(4) = ps;
        d1 = diploop1(xi);
        bx2 = 0.0d0;
        by2 = 0.0d0;
        bz2 = 0.0d0;
        for i = 1:26
            bx2 = bx2 + c1(i).*d1(1,i);
            by2 = by2 + c1(i).*d1(2,i);
            bz2 = bz2 + c1(i).*d1(3,i);
        end
        ss = sqrt((x2-x1).^2+(y2-y1).^2+(z2-z1).^2);
        ds = sqrt((x-x1).^2+(y-y1).^2+(z-z1).^2);
        frac = ds./ss;
        
        bx = bx1.*(1.0d0-frac)+bx2.*frac;
        by = by1.*(1.0d0-frac)+by2.*frac;
        bz = bz1.*(1.0d0-frac)+bz2.*frac;
    end

    [bsx, bsy, bsz] = birk1shld(ps, x, y, z);
    bx = bx + bsx;
    by = by + bsy;
    bz = bz + bsz;
end

function d = diploop1(xi)
    global xx1 yy1 dipx dipy rh dr xcentre radius tilt
    xx=xx1;
    yy=yy1;

    x = xi(1);
    y = xi(2);
    z = xi(3);
    ps= xi(4);
    sps=sin(ps);
    
    d = zeros(3, 26);

    for i = 1:12
        r2 = (xx(i).*dipx).^2+(yy(i).*dipy).^2;
        r  = sqrt(r2);
        rmrh = r - rh;
        rprh = r + rh;
        dr2 = dr.^2;
        sqm = sqrt(rmrh.^2+dr2);
        sqp = sqrt(rprh.^2+dr2);
        c = sqp - sqm;
        q = sqrt((rh+1.0d0).^2+dr2)-sqrt((rh-1.0d0).^2+dr2);
        
        spsas = sps./r.*c./q;
        cpsas = sqrt(1.0d0-spsas.^2);
        
        xd = (xx(i).*dipx).*cpsas;
        yd = (yy(i).*dipy);
        zd = -(xx(i).*dipx).*spsas;
        
        [bx1x, by1x, bz1x, ~, ~, ~, bx1z, by1z, bz1z] = dipxyz(x-xd, y-yd, z-zd);
        if abs(yd) > 1.0d-10
            [bx2x, by2x, bz2x, ~, ~, ~, bx2z, by2z, bz2z] = dipxyz(x-xd, y+yd, z-zd);
        else
            bx2x = 0.0d0;
            by2x = 0.0d0;
            bz2x = 0.0d0;
            bx2z = 0.0d0;
            by2z = 0.0d0;
            bz2z = 0.0d0;
        end

        d(1,i) = bx1z + bx2z;
        d(2,i) = by1z + by2z;
        d(3,i) = bz1z + bz2z;
        d(1,i+12) = (bx1x+bx2x).*sps;
        d(2,i+12) = (by1x+by2x).*sps;
        d(3,i+12) = (bz1x+bz2x).*sps;
    end

    r2 = (xcentre(1)+radius(1)).^2;
    r  = sqrt(r2);
    rmrh = r - rh;
    rprh = r + rh;
    dr2 = dr.^2;
    sqm = sqrt(rmrh.^2+dr2);
    sqp = sqrt(rprh.^2+dr2);
    c = sqp - sqm;
    q = sqrt((rh+1.0d0).^2+dr2)-sqrt((rh-1.0d0).^2+dr2);
    
    spsas = sps./r.*c./q;
    cpsas = sqrt(1.0d0-spsas.^2);
    
    xoct1 = x.*cpsas - z.*spsas;
    yoct1 = y;
    zoct1 = x.*spsas + z.*cpsas;
    
    [bxoct1, byoct1, bzoct1] = crosslp(xoct1, yoct1, zoct1, xcentre(1), radius(1), tilt);
    
    d(1,25) = bxoct1.*cpsas + bzoct1.*spsas;
    d(2,25) = byoct1;
    d(3,25) = -bxoct1.*spsas + bzoct1.*cpsas;

    r2 = (radius(2)-xcentre(2)).^2;
    r  = sqrt(r2);
    rmrh = r - rh;
    rprh = r + rh;
    dr2  = dr.^2;
    sqm = sqrt(rmrh.^2+dr2);
    sqp = sqrt(rprh.^2+dr2);
    c = sqp - sqm;
    q = sqrt((rh+1.0d0).^2+dr2)-sqrt((rh-1.0d0).^2+dr2);
    
    spsas = sps./r.*c./q;
    cpsas = sqrt(1.0d0-spsas.^2);
    
    xoct2 = x.*cpsas - z.*spsas - xcentre(2);
    yoct2 = y;
    zoct2 = x.*spsas + z.*cpsas;
    
    [bx, by, bz] = circle(xoct2, yoct2, zoct2, radius(2));
    
    d(1,26) =  bx.*cpsas + bz.*spsas;
    d(2,26) =  by;
    d(3,26) = -bx.*spsas + bz.*cpsas;
end

function [bx, by, bz] = circle(x, y, z, rl)
    rho2 = x.*x+y.*y;
    rho  = sqrt(rho2);
    r22  = z.*z+(rho+rl).^2;
    r2   = sqrt(r22);
    r12  = r22-4.0d0.*rho.*rl;
    r32  = 0.5d0.*(r12+r22);
    xk2  = 1.0d0-r12./r22;
    xk2s = 1.0d0-xk2;
    dl   = log(1.0d0./xk2s);

    k = 1.38629436112d0+xk2s.*(0.09666344259d0+xk2s.*(0.03590092383+ ...
         xk2s.*(0.03742563713+xk2s.*0.01451196212))) +dl.* ...
        (0.5d0+xk2s.*(0.12498593597d0+xk2s.*(0.06880248576d0+ ...
         xk2s.*(0.03328355346d0+xk2s.*0.00441787012d0))));

    e = 1.0d0+xk2s.*(0.44325141463d0+xk2s.*(0.0626060122d0+xk2s.* ...
        (0.04757383546d0+xk2s.*0.01736506451d0))) +dl.* ...
        xk2s.*(0.2499836831d0+xk2s.*(0.09200180037d0+xk2s.* ...
        (0.04069697526d0+xk2s.*0.00526449639d0)));

    if rho > 1.0d-6
        brho = z./(rho2.*r2).*(r32./r12.*e-k);
    else
        brho = pi.*rl./r2.*(rl-rho)./r12.*z./(r32-rho2);
    end

    bx = brho.*x;
    by = brho.*y;
    bz = (k-e.*(r32-2.0d0.*rl.*rl)./r12)./r2;
end

function [bx, by, bz] = crosslp(x, y, z, xc, rl, al)
    cal = cos(al);
    sal = sin(al);

    y1 = y.*cal - z.*sal;
    z1 = y.*sal + z.*cal;
    y2 = y.*cal + z.*sal;
    z2 = -y.*sal + z.*cal;
    
    [bx1, by1, bz1] = circle(x-xc, y1, z1, rl);
    [bx2, by2, bz2] = circle(x-xc, y2, z2, rl);
    
    bx = bx1+bx2;
    by = (by1+by2).*cal+(bz1-bz2).*sal;
    bz = -(by1-by2).*sal+(bz1+bz2).*cal;
end

function [bxx, byx, bzx, bxy, byy, bzy, bxz, byz, bzz] = dipxyz(x, y, z)
    x2 = x.^2;
    y2 = y.^2;
    z2 = z.^2;
    r2 = x2+y2+z2;

    xmr5  = 30574.0d0./(r2.*r2.*sqrt(r2));
    xmr53 = 3.0d0.*xmr5;
    
    bxx = xmr5.*(3.0d0.*x2-r2);
    byx = xmr53.*x.*y;
    bzx = xmr53.*x.*z;

    bxy = byx;
    byy = xmr5.*(3.0d0.*y2-r2);
    bzy = xmr53.*y.*z;

    bxz = bzx;
    byz = bzy;
    bzz = xmr5.*(3.0d0.*z2-r2);
end

function d = condip1(xi)
    global dx scalein scaleout xx2 yy2 zz2
    xx = xx2;
    yy = yy2;
    zz = zz2;

    x = xi(1);
    y = xi(2);
    z = xi(3);
    ps= xi(4);
    sps=sin(ps);
    cps=cos(ps);

    xsm = x.*cps - z.*sps - dx;
    zsm = z.*cps + x.*sps;
    ro2 = xsm.^2 + y.^2;
    ro  = sqrt(ro2);

    cf(1) = xsm./ro;
    sf(1) = y./ro;
    cf(2) = cf(1).^2-sf(1).^2;
    sf(2) = 2.*sf(1).*cf(1);
    cf(3) = cf(2).*cf(1) - sf(2).*sf(1);
    sf(3) = sf(2).*cf(1) + cf(2).*sf(1);
    cf(4) = cf(3).*cf(1) - sf(3).*sf(1);
    sf(4) = sf(3).*cf(1) + cf(3).*sf(1);
    cf(5) = cf(4).*cf(1) - sf(4).*sf(1);
    sf(5) = sf(4).*cf(1) + cf(4).*sf(1);

    r2 = ro2 + zsm.^2;
    r  = sqrt(r2);
    c  = zsm./r;
    s  = ro./r;
    ch = sqrt(0.5d0.*(1.0d0+c));
    sh = sqrt(0.5d0.*(1.0d0-c));
    tnh= sh./ch;
    cnh= 1.0d0./tnh;

    d = zeros(3, 79);
    
    for m = 1:5
        bt = m.*cf(m)./(r.*s).*(tnh.^m+cnh.^m);
        bf = -0.5d0.*m.*sf(m)./r.*(tnh.^(m-1)./ch.^2-cnh.^(m-1)./sh.^2);
        bxsm = bt.*c.*cf(1)-bf.*sf(1);
        by = bt.*c.*sf(1)+bf.*cf(1);
        bzsm = -bt.*s;

        d(1,m) = bxsm.*cps + bzsm.*sps;
        d(2,m) = by;
        d(3,m) = -bxsm.*sps + bzsm.*cps;
    end

    xsm = x.*cps - z.*sps;
    zsm = z.*cps + x.*sps;

    for i = 1:9
        if i==3 || i==5 || i==6
            xd =  xx(i).*scalein;
            yd =  yy(i).*scalein;
        else
            xd =  xx(i).*scaleout;
            yd =  yy(i).*scaleout;
        end

        zd =  zz(i);

        [bx1x, by1x, bz1x, bx1y, by1y, bz1y, bx1z, by1z, bz1z] = dipxyz(xsm-xd, y-yd, zsm-zd);
        [bx2x, by2x, bz2x, bx2y, by2y, bz2y, bx2z, by2z, bz2z] = dipxyz(xsm-xd, y+yd, zsm-zd);
        [bx3x, by3x, bz3x, bx3y, by3y, bz3y, bx3z, by3z, bz3z] = dipxyz(xsm-xd, y-yd, zsm+zd);
        [bx4x, by4x, bz4x, bx4y, by4y, bz4y, bx4z, by4z, bz4z] = dipxyz(xsm-xd, y+yd, zsm+zd);

        ix = i.*3 + 3;
        iy = ix + 1;
        iz = iy + 1;

        d(1,ix) = (bx1x+bx2x-bx3x-bx4x).*cps + (bz1x+bz2x-bz3x-bz4x).*sps;
        d(2,ix) =  by1x+by2x-by3x-by4x;
        d(3,ix) = (bz1x+bz2x-bz3x-bz4x).*cps - (bx1x+bx2x-bx3x-bx4x).*sps;

        d(1,iy) = (bx1y-bx2y-bx3y+bx4y).*cps + (bz1y-bz2y-bz3y+bz4y).*sps;
        d(2,iy) =  by1y-by2y-by3y+by4y;
        d(3,iy) = (bz1y-bz2y-bz3y+bz4y).*cps - (bx1y-bx2y-bx3y+bx4y).*sps;

        d(1,iz) = (bx1z+bx2z+bx3z+bx4z).*cps + (bz1z+bz2z+bz3z+bz4z).*sps;
        d(2,iz) =  by1z+by2z+by3z+by4z;
        d(3,iz) = (bz1z+bz2z+bz3z+bz4z).*cps - (bx1z+bx2z+bx3z+bx4z).*sps;

        ix = ix + 27;
        iy = iy + 27;
        iz = iz + 27;

        d(1,ix) = sps.*((bx1x+bx2x+bx3x+bx4x).*cps + (bz1x+bz2x+bz3x+bz4x).*sps);
        d(2,ix) = sps.*(by1x+by2x+by3x+by4x);
        d(3,ix) = sps.*((bz1x+bz2x+bz3x+bz4x).*cps - (bx1x+bx2x+bx3x+bx4x).*sps);

        d(1,iy) = sps.*((bx1y-bx2y+bx3y-bx4y).*cps + (bz1y-bz2y+bz3y-bz4y).*sps);
        d(2,iy) = sps.*(by1y-by2y+by3y-by4y);
        d(3,iy) = sps.*((bz1y-bz2y+bz3y-bz4y).*cps - (bx1y-bx2y+bx3y-bx4y).*sps);

        d(1,iz) = sps.*((bx1z+bx2z-bx3z-bx4z).*cps + (bz1z+bz2z-bz3z-bz4z).*sps);
        d(2,iz) = sps.*(by1z+by2z-by3z-by4z);
        d(3,iz) = sps.*((bz1z+bz2z-bz3z-bz4z).*cps - (bx1z+bx2z-bx3z-bx4z).*sps);
    end

    for i = 1:5
        zd = zz(i+9);
        [bx1x, by1x, bz1x, ~, ~, ~, bx1z, by1z, bz1z] = dipxyz(xsm, y, zsm-zd);
        [bx2x, by2x, bz2x, ~, ~, ~, bx2z, by2z, bz2z] = dipxyz(xsm, y, zsm+zd);
        
        ix = 58 + i.*2;
        iz = ix + 1;
        
        d(1,ix) = (bx1x-bx2x).*cps + (bz1x-bz2x).*sps;
        d(2,ix) =  by1x-by2x;
        d(3,ix) = (bz1x-bz2x).*cps - (bx1x-bx2x).*sps;

        d(1,iz) = (bx1z+bx2z).*cps + (bz1z+bz2z).*sps;
        d(2,iz) =  by1z+by2z;
        d(3,iz) = (bz1z+bz2z).*cps - (bx1z+bx2z).*sps;

        ix = ix + 10;
        iz = iz + 10;
        
        d(1,ix) = sps.*((bx1x+bx2x).*cps + (bz1x+bz2x).*sps);
        d(2,ix) = sps.*(by1x+by2x);
        d(3,ix) = sps.*((bz1x+bz2x).*cps - (bx1x+bx2x).*sps);

        d(1,iz) = sps.*((bx1z-bx2z).*cps + (bz1z-bz2z).*sps);
        d(2,iz) = sps.*(by1z-by2z);
        d(3,iz) = sps.*((bz1z-bz2z).*cps - (bx1z-bx2z).*sps);
    end
end

function [bx,by,bz] = birk1shld(ps,x,y,z)
    persistent a
    if isempty(a)
        a = [1.174198045 -1.463820502 4.840161537 -3.674506864 ...
          82.18368896 -94.94071588 -4122.331796 4670.278676 -21.54975037 ...
          26.72661293 -72.81365728 44.09887902 40.08073706 -51.23563510 ...
          1955.348537 -1940.971550 794.0496433 -982.2441344 1889.837171 ...
          -558.9779727 -1260.543238 1260.063802 -293.5942373 344.7250789 ...
          -773.7002492 957.0094135 -1824.143669 520.7994379 1192.484774 ...
          -1192.184565 89.15537624 -98.52042999 -0.8168777675E-01 ...
          0.4255969908E-01 0.3155237661 -0.3841755213 2.494553332 ...
          -0.6571440817E-01 -2.765661310 0.4331001908 0.1099181537 ...
          -0.6154126980E-01 -0.3258649260 0.6698439193 -5.542735524 ...
          0.1604203535 5.854456934 -0.8323632049 3.732608869 -3.130002153 ...
          107.0972607 -32.28483411 -115.2389298 54.45064360 -0.5826853320 ...
          -3.582482231 -4.046544561 3.311978102 -104.0839563 30.26401293 ...
          97.29109008 -50.62370872 -296.3734955 127.7872523 5.303648988 ...
          10.40368955 69.65230348 466.5099509 1.645049286 3.825838190 ...
          11.66675599 558.9781177 1.826531343 2.066018073 25.40971369 ...
          990.2795225 2.319489258 4.555148484 9.691185703 591.8280358];
    end
    
    p1 = a(65:68);
    r1 = a(69:72);
    q1 = a(73:76);
    s1 = a(77:80);

    bx = 0.0d0;
    by = 0.0d0;
    bz = 0.0d0;
    
    cps = cos(ps);
    sps = sin(ps);
    s3ps = 4.0d0.*cps.^2 - 1.0d0;

    rp = zeros(1,4);
    rr = zeros(1,4);
    rq = zeros(1,4);
    rs = zeros(1,4);
    for i = 1:4
        rp(i) = 1.0d0./p1(i);
        rr(i) = 1.0d0./r1(i);
        rq(i) = 1.0d0./q1(i);
        rs(i) = 1.0d0./s1(i);
    end

    l = 0;

    for m = 1:2

        for i = 1:4
            cypi = cos(y.*rp(i));
            cyqi = cos(y.*rq(i));
            sypi = sin(y.*rp(i));
            syqi = sin(y.*rq(i));

            for k = 1:4
                szrk = sin(z.*rr(k));
                czsk = cos(z.*rs(k));
                czrk = cos(z.*rr(k));
                szsk = sin(z.*rs(k));
                sqpr = sqrt(rp(i).^2+rr(k).^2);
                sqqs = sqrt(rq(i).^2+rs(k).^2);
                epr  = exp(x.*sqpr);
                eqs  = exp(x.*sqqs);

                for n = 1:2
                    if m == 1
                        if n == 1
                            hx = -sqpr.*epr.*cypi.*szrk;
                            hy = rp(i).*epr.*sypi.*szrk;
                            hz = -rr(k).*epr.*cypi.*czrk;
                        else
                            hx = hx.*cps;
                            hy = hy.*cps;
                            hz = hz.*cps;
                        end
                    else
                        if n == 1
                            hx = -sps.*sqqs.*eqs.*cyqi.*czsk;
                            hy = sps.*rq(i).*eqs.*syqi.*czsk;
                            hz = sps.*rs(k).*eqs.*cyqi.*szsk;
                        else
                            hx = hx.*s3ps;
                            hy = hy.*s3ps;
                            hz = hz.*s3ps;
                        end
                    end
                    
                    l = l+1;

                    bx = bx + a(l).*hx;
                    by = by + a(l).*hy;
                    bz = bz + a(l).*hz;
                end
            end
        end
    end
    
end

function [bx, by, bz] = birk2tot_02(ps, x, y, z)
    [wx, wy, wz] = birk2shl(x, y, z, ps);
    [hx, hy, hz] = r2_birk( x, y, z, ps);

    bx = wx + hx;
    by = wy + hy;
    bz = wz + hz;
end

function [hx, hy, hz] = birk2shl(x, y, z, ps)
    persistent a
    if isempty(a)
        a = [-111.6371348 124.5402702 110.3735178 -122.0095905 ...
              111.9448247 -129.1957743 -110.7586562 126.5649012 -0.7865034384 ...
              -0.2483462721 0.8026023894 0.2531397188 10.72890902 0.8483902118 ...
              -10.96884315 -0.8583297219 13.85650567 14.90554500 10.21914434 ...
              10.09021632 6.340382460 14.40432686 12.71023437 12.83966657];
    end
    
    p = a(17:18);
    r = a(19:20);
    q = a(21:22);
    s = a(23:24);

    cps = cos(ps);
    sps = sin(ps);
    s3ps= 4.0d0.*cps.^2-1.0d0;

    l = 0;
    hx =0.0d0;
    hy =0.0d0;
    hz =0.0d0;
    
    for m = 1:2
        for i = 1:2
            cypi = cos(y./p(i));
            cyqi = cos(y./q(i));
            sypi = sin(y./p(i));
            syqi = sin(y./q(i));

            for k = 1:2
                szrk = sin(z./r(k));
                czsk = cos(z./s(k));
                czrk = cos(z./r(k));
                szsk = sin(z./s(k));
                sqpr = sqrt(1.0d0./p(i).^2+1.0d0./r(k).^2);
                sqqs = sqrt(1.0d0./q(i).^2+1.0d0./s(k).^2);
                epr  = exp(x.*sqpr);
                eqs  = exp(x.*sqqs);

                for n = 1:2
                    l = l+1;
                    
                    if m == 1
                        if n == 1
                            dx = -sqpr.*epr.*cypi.*szrk;
                            dy = epr./p(i).*sypi.*szrk;
                            dz = -epr./r(k).*cypi.*czrk;
                            hx = hx + a(l).*dx;
                            hy = hy + a(l).*dy;
                            hz = hz + a(l).*dz;
                        else
                            dx = dx.*cps;
                            dy = dy.*cps;
                            dz = dz.*cps;
                            hx = hx + a(l).*dx;
                            hy = hy + a(l).*dy;
                            hz = hz + a(l).*dz;
                        end
                    else
                        if n == 1
                            dx = -sps.*sqqs.*eqs.*cyqi.*czsk;
                            dy = sps.*eqs./q(i).*syqi.*czsk;
                            dz = sps.*eqs./s(k).*cyqi.*szsk;
                            hx = hx + a(l).*dx;
                            hy = hy + a(l).*dy;
                            hz = hz + a(l).*dz;
                        else
                            dx = dx.*s3ps;
                            dy = dy.*s3ps;
                            dz = dz.*s3ps;
                            hx = hx + a(l).*dx;
                            hy = hy + a(l).*dy;
                            hz = hz + a(l).*dz;
                        end
                    end
                end
            end
        end
    end
end

function [bx,by,bz] = r2_birk(x,y,z,ps)
    delarg = 0.030D0;
    delarg1 = 0.015D0;

    cps = cos(ps);
    sps = sin(ps);

    xsm = x.*cps - z.*sps;
    zsm = z.*cps + x.*sps;

    xks = xksi(xsm, y, zsm);
    
    if xks < -(delarg + delarg1)
        [bxsm, by, bzsm] = r2outer(xsm, y, zsm);
        bxsm = -bxsm.*0.02;
        by   = -by.*0.02;
        bzsm = -bzsm.*0.02;
    end
    
    if xks >= -(delarg + delarg1) && xks < -delarg + delarg1
        [bxsm1, by1, bzsm1] = r2outer(xsm, y, zsm);
        [bxsm2, by2, bzsm2] = r2sheet(xsm, y, zsm);
        f2 = -0.02.*tksi(xks,-delarg,delarg1);
        f1 = -0.02-f2;
        bxsm = bxsm1.*f1 + bxsm2.*f2;
        by   = by1.*f1 + by2.*f2;
        bzsm = bzsm1.*f1 + bzsm2.*f2;
    end

    if xks >= -delarg + delarg1 && xks < delarg - delarg1
        [bxsm, by, bzsm] = r2sheet(xsm, y, zsm);
        bxsm = -bxsm.*0.02;
        by   = -by.*0.02;
        bzsm = -bzsm.*0.02;
    end

    if xks >= delarg - delarg1 && xks < delarg + delarg1
        [bxsm1, by1, bzsm1] = r2inner(xsm, y, zsm);
        [bxsm2, by2, bzsm2] = r2sheet(xsm, y, zsm);
        f1 =-0.02.*tksi(xks, delarg, delarg1);
        f2 =-0.02-f1;
        bxsm = bxsm1.*f1 + bxsm2.*f2;
        by   = by1.*f1 + by2.*f2;
        bzsm = bzsm1.*f1 + bzsm2.*f2;
    end

    if xks >= delarg + delarg1
        [bxsm, by, bzsm] = r2inner(xsm, y, zsm);
        bxsm = -bxsm.*0.02;
        by   = -by.*0.02;
        bzsm = -bzsm.*0.02;
    end

    bx = bxsm.*cps + bzsm.*sps;
    bz = bzsm.*cps - bxsm.*sps;
end

function [bx, by, bz] = r2inner(x, y, z)
    persistent pl1 pl2 pl3 pl4 pl5 pl6 pl7 pl8
    persistent pn1 pn2 pn3 pn4 pn5 pn6 pn7 pn8
    if isempty(pl1)
        pl1 = 154.185;
        pl2 = -2.12446;
        pl3 = .601735E-01;
        pl4 = -.153954E-02;
        pl5 = .355077E-04;
        pl6 = 29.9996;
        pl7 = 262.886;
        pl8 = 99.9132;

        pn1 = -8.1902;
        pn2 = 6.5239;
        pn3 = 5.504;
        pn4 = 7.7815;
        pn5 = .8573;
        pn6 = 3.0986;
        pn7 = .0774;
        pn8 = -.038;
    end
    
    [cbx, cby, cbz] = bconic(x, y, z, 5);

    [dbx8, dby8, dbz8] = loops4(x, y, z, pn1,pn2,pn3,pn4,pn5,pn6);

    [dbx6, dby6, dbz6] = dipdistr(x-pn7, y, z, 0);
    [dbx7, dby7, dbz7] = dipdistr(x-pn8, y, z, 1);

    bx = pl1.*cbx(1) + pl2.*cbx(2) + pl3.*cbx(3) + pl4.*cbx(4) + pl5.*cbx(5) + pl6.*dbx6 + pl7.*dbx7 + pl8.*dbx8;
    by = pl1.*cby(1) + pl2.*cby(2) + pl3.*cby(3) + pl4.*cby(4) + pl5.*cby(5) + pl6.*dby6 + pl7.*dby7 + pl8.*dby8;
    bz = pl1.*cbz(1) + pl2.*cbz(2) + pl3.*cbz(3) + pl4.*cbz(4) + pl5.*cbz(5) + pl6.*dbz6 + pl7.*dbz7 + pl8.*dbz8;
end

function [cbx, cby, cbz] = bconic(x, y, z, nmax)
    ro2 = x.^2 + y.^2;
    ro  = sqrt(ro2);

    cf  = x./ro;
    sf  = y./ro;
    cfm1= 1.0d0;
    sfm1= 0.0d0;
    tnhm1 = 1.0d0;
    cnhm1 = 1.0d0;
    
    r2 = ro2+z.^2;
    r  = sqrt(r2);
    c  = z./r;
    s  = ro./r;
    ch = sqrt(0.5d0.*(1.0d0+c));
    sh = sqrt(0.5d0.*(1.0d0-c));
    tnh= sh./ch;
    cnh= 1.0d0./tnh;

    cbx = zeros(nmax, 1);
    cby = zeros(nmax, 1);
    cbz = zeros(nmax, 1);
    
    for m = 1:nmax
        cfm  = cfm1.*cf - sfm1.*sf;
        sfm  = cfm1.*sf + sfm1.*cf;
        cfm1 = cfm;
        sfm1 = sfm;
        tnhm = tnhm1.*tnh;
        cnhm = cnhm1.*cnh;
        bt   = m.*cfm./(r.*s).*(tnhm+cnhm);
        bf   = -0.5d0.*m.*sfm./r.*(tnhm1./ch.^2-cnhm1./sh.^2);
        tnhm1= tnhm;
        cnhm1= cnhm;
        
        cbx(m) = bt.*c.*cf - bf.*sf;
        cby(m) = bt.*c.*sf + bf.*cf;
        cbz(m) = -bt.*s;
    end
end

function [bx, by, bz] = dipdistr(x, y, z, mode)
    x2   = x.*x;
    rho2 = x2 + y.*y;
    r2   = rho2 + z.*z;
    r3   = r2.*sqrt(r2);

    if mode == 0
        bx = z./rho2.^2.*(r2.*(y.*y-x2)-rho2.*x2)./r3;
        by = -x.*y.*z./rho2.^2.*(2.0d0.*r2+rho2)./r3;
        bz = x./r3;
    else
        bx = z./rho2.^2.*(y.*y-x2);
        by = -2.0d0.*x.*y.*z./rho2.^2;
        bz = x./rho2;
    end
end

function [bx, by, bz] = r2outer(x, y, z)
    persistent pl1 pl2 pl3 pl4 pl5
    persistent pn1 pn2 pn3 pn4 pn5 pn6 pn7 pn8 pn9 
    persistent pn10 pn11 pn12 pn13 pn14 pn15 pn16 pn17
    if isempty(pl1)
        pl1 = -34.105;
        pl2 = -2.00019;
        pl3 = 628.639;
        pl4 = 73.4847;
        pl5 = 12.5162;

        pn1 = .55;
        pn2 = .694;
        pn3 = .0031;
        pn4 = 1.55;
        pn5 = 2.8;
        pn6 = .1375;
        pn7 = -.7;
        pn8 = .2;
        pn9 = .9625;
        pn10 = -2.994;
        pn11 = 2.925;
        pn12 = -1.775;
        pn13 = 4.3;
        pn14 = -.275;
        pn15 = 2.7;
        pn16 = .4312;
        pn17 = 1.55;
    end
     
    [dbx1, dby1, dbz1] = crosslp(x, y, z, pn1, pn2, pn3);
    [dbx2, dby2, dbz2] = crosslp(x, y, z, pn4, pn5, pn6);
    [dbx3, dby3, dbz3] = crosslp(x, y, z, pn7, pn8, pn9);

    [dbx4, dby4, dbz4] = circle(x-pn10, y, z, pn11);

    [dbx5, dby5, dbz5] = loops4(x, y, z, pn12, pn13, pn14, pn15, pn16, pn17);

    bx = pl1.*dbx1 + pl2.*dbx2 + pl3.*dbx3 + pl4.*dbx4 + pl5.*dbx5;
    by = pl1.*dby1 + pl2.*dby2 + pl3.*dby3 + pl4.*dby4 + pl5.*dby5;
    bz = pl1.*dbz1 + pl2.*dbz2 + pl3.*dbz3 + pl4.*dbz4 + pl5.*dbz5;
end

function [bx, by, bz] = loops4(x, y, z, xc, yc, zc, r, theta, phi)
    ct = cos(theta);
    st = sin(theta);
    cp = cos(phi);
    sp = sin(phi);

    xs  = (x-xc).*cp + (y-yc).*sp;
    yss = (y-yc).*cp - (x-xc).*sp;
    zs  = z - zc;
    xss = xs.*ct - zs.*st;
    zss = zs.*ct + xs.*st;

    [bxss, bys, bzss] = circle(xss, yss, zss, r);
    bxs = bxss.*ct + bzss.*st;
    bz1 = bzss.*ct - bxss.*st;
    bx1 = bxs.*cp - bys.*sp;
    by1 = bxs.*sp + bys.*cp;

    xs  =( x-xc).*cp - (y+yc).*sp;
    yss = (y+yc).*cp + (x-xc).*sp;
    zs  = z - zc;
    xss = xs.*ct - zs.*st;
    zss = zs.*ct + xs.*st;

    [bxss, bys, bzss] = circle(xss, yss, zss, r);
    bxs = bxss.*ct + bzss.*st;
    bz2 = bzss.*ct - bxss.*st;
    bx2 = bxs.*cp + bys.*sp;
    by2 = -bxs.*sp + bys.*cp;

    xs  = -(x-xc).*cp + (y+yc).*sp;
    yss = -(y+yc).*cp - (x-xc).*sp;
    zs  = z + zc;
    xss = xs.*ct - zs.*st;
    zss = zs.*ct + xs.*st;

    [bxss, bys, bzss] = circle(xss, yss, zss, r);
    bxs = bxss.*ct + bzss.*st;
    bz3 = bzss.*ct - bxss.*st;
    bx3 = -bxs.*cp - bys.*sp;
    by3 = bxs.*sp - bys.*cp;

    xs  = -(x-xc).*cp - (y-yc).*sp;
    yss = -(y-yc).*cp + (x-xc).*sp;
    zs  = z + zc;
    xss = xs.*ct - zs.*st;
    zss = zs.*ct + xs.*st;

    [bxss, bys, bzss] = circle(xss, yss, zss, r);
    bxs = bxss.*ct + bzss.*st;
    bz4 = bzss.*ct - bxss.*st;
    bx4 = -bxs.*cp + bys.*sp;
    by4 = -bxs.*sp - bys.*cp;

    bx = bx1 + bx2 + bx3 + bx4;
    by = by1 + by2 + by3 + by4;
    bz = bz1 + bz2 + bz3 + bz4;
end

function [bx, by, bz] = r2sheet(x, y, z)
    persistent pnonx pnony pnonz a b c
    if isempty(pnonx)
        pnonx = [-19.0969D0,-9.28828D0,-0.129687D0,5.58594D0, ...
            22.5055D0,0.483750D-01,0.396953D-01,0.579023D-01];
        pnony = [-13.6750D0,-6.70625D0,2.31875D0,11.4062D0, ...
            20.4562D0,0.478750D-01,0.363750D-01,0.567500D-01];
        pnonz = [-16.7125D0,-16.4625D0,-0.1625D0,5.1D0, ...
            23.7125D0,0.355625D-01,0.318750D-01,0.538750D-01];

        a = [8.07190D0 -7.39582D0 -7.62341D0 0.684671D0 -13.5672D0 11.6681D0 ...
              13.1154 -0.890217D0 7.78726D0 -5.38346D0 -8.08738D0 0.609385D0 ...
              -2.70410D0  3.53741D0 3.15549D0 -1.11069D0 -8.47555D0 0.278122D0 ...
               2.73514D0 4.55625D0 13.1134D0 1.15848D0 -3.52648D0 -8.24698D0 ...
              -6.85710D0 -2.81369D0  2.03795D0  4.64383D0 2.49309D0 -1.22041D0 ...
              -1.67432D0 -0.422526D0 -5.39796D0 7.10326D0 5.53730D0 -13.1918D0 ...
               4.67853D0 -7.60329D0 -2.53066D0  7.76338D0  5.60165D0 5.34816D0 ...
              -4.56441D0 7.05976D0 -2.62723D0 -0.529078D0 1.42019D0 -2.93919D0 ...
               55.6338D0 -1.55181D0 39.8311D0 -80.6561D0 -46.9655D0 32.8925D0 ...
              -6.32296D0 19.7841D0 124.731D0 10.4347D0 -30.7581D0 102.680D0 ...
              -47.4037D0 -3.31278D0 9.37141D0 -50.0268D0 -533.319D0 110.426D0 ...
               1000.20D0 -1051.40D0  1619.48D0 589.855D0 -1462.73D0 1087.10D0 ...
               -1994.73D0 -1654.12D0 1263.33D0 -260.210D0 1424.84D0 1255.71D0 ...
               -956.733D0  219.946D0];

        b = [-9.08427D0 10.6777D0 10.3288D0 -0.969987D0 6.45257D0 -8.42508D0 ...
              -7.97464D0 1.41996D0 -1.92490D0 3.93575D0 2.83283D0 -1.48621D0 ...
             0.244033D0 -0.757941D0 -0.386557D0 0.344566D0 9.56674D0 -2.5365D0 ...
              -3.32916D0 -5.86712D0 -6.19625D0 1.83879D0 2.52772D0 4.34417D0 ...
              1.87268D0 -2.13213D0 -1.69134D0 -.176379D0 -.261359D0 .566419D0 ...
              0.3138D0 -0.134699D0 -3.83086D0 -8.4154D0 4.77005D0 -9.31479D0 ...
              37.5715D0 19.3992D0 -17.9582D0 36.4604D0 -14.9993D0 -3.1442D0 ...
              6.17409D0 -15.5519D0 2.28621D0 -0.891549D-2 -.462912D0 2.47314D0 ...
              41.7555D0 208.614D0 -45.7861D0 -77.8687D0 239.357D0 -67.9226D0 ...
              66.8743D0 238.534D0 -112.136D0 16.2069D0 -40.4706D0 -134.328D0 ...
              21.56D0 -0.201725D0 2.21D0 32.5855D0 -108.217D0 -1005.98D0 ...
              585.753D0 323.668D0 -817.056D0 235.750D0 -560.965D0 -576.892D0 ...
              684.193D0 85.0275D0 168.394D0 477.776D0 -289.253D0 -123.216D0 ...
              75.6501D0 -178.605D0];

          c = [1167.61D0 -917.782D0 -1253.2D0 -274.128D0 -1538.75D0 1257.62D0 ...
              1745.07D0 113.479D0 393.326D0 -426.858D0 -641.1D0 190.833D0 ...
              -29.9435D0 -1.04881D0 117.125D0 -25.7663D0 -1168.16D0 910.247D0 ...
              1239.31D0 289.515D0 1540.56D0 -1248.29D0 -1727.61D0 -131.785D0 ...
              -394.577D0 426.163D0 637.422D0 -187.965D0 30.0348D0 0.221898D0 ...
              -116.68D0 26.0291D0 12.6804D0 4.84091D0 1.18166D0 -2.75946D0 ...
              -17.9822D0 -6.80357D0 -1.47134D0 3.02266D0 4.79648D0 0.665255D0 ...
              -0.256229D0 -0.857282D-1 -0.588997D0 0.634812D-1 0.164303D0 ...
              -0.15285D0 22.2524D0 -22.4376D0 -3.85595D0 6.07625D0 -105.959D0 ...
              -41.6698D0 0.378615D0 1.55958D0 44.3981D0 18.8521D0 3.19466D0 ...
               5.89142D0 -8.63227D0 -2.36418D0 -1.027D0 -2.31515D0 1035.38D0 ...
               2040.66D0 -131.881D0 -744.533D0 -3274.93D0 -4845.61D0 482.438D0 ...
              1567.43D0 1354.02D0 2040.47D0 -151.653D0 -845.012D0 -111.723D0 ...
              -265.343D0 -26.1171D0 216.632D0];
    end
     
    xks = xksi(x,y,z);
    t1x = xks./sqrt(xks.^2+pnonx(6).^2);
    t2x = pnonx(7).^3./sqrt(xks.^2+pnonx(7).^2).^3;
    t3x = xks./sqrt(xks.^2+pnonx(8).^2).^5 .*3.493856d0.*pnonx(8).^4;

    t1y = xks./sqrt(xks.^2+pnony(6).^2);
    t2y = pnony(7).^3./sqrt(xks.^2+pnony(7).^2).^3;
    t3y = xks./sqrt(xks.^2+pnony(8).^2).^5 .*3.493856d0.*pnony(8).^4;

    t1z = xks./sqrt(xks.^2+pnonz(6).^2);
    t2z = pnonz(7).^3./sqrt(xks.^2+pnonz(7).^2).^3;
    t3z = xks./sqrt(xks.^2+pnonz(8).^2).^5 .*3.493856d0.*pnonz(8).^4;

    rho2 = x.*x+y.*y;
    r    = sqrt(rho2+z.*z);
    rho  = sqrt(rho2);

    c1p = x./rho;
    s1p = y./rho;
    s2p = 2.0d0.*s1p.*c1p;
    c2p = c1p.*c1p - s1p.*s1p;
    s3p = s2p.*c1p + c2p.*s1p;
    c3p = c2p.*c1p - s2p.*s1p;
    s4p = s3p.*c1p + c3p.*s1p;
    ct = z./r;

    s1 = fexp(ct, pnonx(1));
    s2 = fexp(ct, pnonx(2));
    s3 = fexp(ct, pnonx(3));
    s4 = fexp(ct, pnonx(4));
    s5 = fexp(ct, pnonx(5));

    bx = s1.*((a(1)+a(2).*t1x+a(3).*t2x+a(4).*t3x) ...
            +c1p.*(a(5)+a(6).*t1x+a(7).*t2x+a(8).*t3x) ...
            +c2p.*(a(9)+a(10).*t1x+a(11).*t2x+a(12).*t3x) ...
            +c3p.*(a(13)+a(14).*t1x+a(15).*t2x+a(16).*t3x)) ...
        +s2.*((a(17)+a(18).*t1x+a(19).*t2x+a(20).*t3x) ...
            +c1p.*(a(21)+a(22).*t1x+a(23).*t2x+a(24).*t3x) ...
            +c2p.*(a(25)+a(26).*t1x+a(27).*t2x+a(28).*t3x) ...
            +c3p.*(a(29)+a(30).*t1x+a(31).*t2x+a(32).*t3x)) ...
        +s3.*((a(33)+a(34).*t1x+a(35).*t2x+a(36).*t3x) ...
            +c1p.*(a(37)+a(38).*t1x+a(39).*t2x+a(40).*t3x) ...
            +c2p.*(a(41)+a(42).*t1x+a(43).*t2x+a(44).*t3x) ...
            +c3p.*(a(45)+a(46).*t1x+a(47).*t2x+a(48).*t3x)) ...
        +s4.*((a(49)+a(50).*t1x+a(51).*t2x+a(52).*t3x) ...
            +c1p.*(a(53)+a(54).*t1x+a(55).*t2x+a(56).*t3x) ...
            +c2p.*(a(57)+a(58).*t1x+a(59).*t2x+a(60).*t3x) ...
            +c3p.*(a(61)+a(62).*t1x+a(63).*t2x+a(64).*t3x)) ...
        +s5.*((a(65)+a(66).*t1x+a(67).*t2x+a(68).*t3x) ...
            +c1p.*(a(69)+a(70).*t1x+a(71).*t2x+a(72).*t3x) ...
            +c2p.*(a(73)+a(74).*t1x+a(75).*t2x+a(76).*t3x) ...
            +c3p.*(a(77)+a(78).*t1x+a(79).*t2x+a(80).*t3x));

    s1 = fexp(ct, pnony(1));
    s2 = fexp(ct, pnony(2));
    s3 = fexp(ct, pnony(3));
    s4 = fexp(ct, pnony(4));
    s5 = fexp(ct, pnony(5));

    by = s1.*(s1p.*(b(1)+b(2).*t1y+b(3).*t2y+b(4).*t3y) ...
          +s2p.*(b(5)+b(6).*t1y+b(7).*t2y+b(8).*t3y) ...
          +s3p.*(b(9)+b(10).*t1y+b(11).*t2y+b(12).*t3y) ...
          +s4p.*(b(13)+b(14).*t1y+b(15).*t2y+b(16).*t3y)) ...
      +s2.*(s1p.*(b(17)+b(18).*t1y+b(19).*t2y+b(20).*t3y) ...
          +s2p.*(b(21)+b(22).*t1y+b(23).*t2y+b(24).*t3y) ...
          +s3p.*(b(25)+b(26).*t1y+b(27).*t2y+b(28).*t3y) ...
          +s4p.*(b(29)+b(30).*t1y+b(31).*t2y+b(32).*t3y)) ...
      +s3.*(s1p.*(b(33)+b(34).*t1y+b(35).*t2y+b(36).*t3y) ...
          +s2p.*(b(37)+b(38).*t1y+b(39).*t2y+b(40).*t3y) ...
          +s3p.*(b(41)+b(42).*t1y+b(43).*t2y+b(44).*t3y) ...
          +s4p.*(b(45)+b(46).*t1y+b(47).*t2y+b(48).*t3y)) ...
      +s4.*(s1p.*(b(49)+b(50).*t1y+b(51).*t2y+b(52).*t3y) ...
          +s2p.*(b(53)+b(54).*t1y+b(55).*t2y+b(56).*t3y) ...
          +s3p.*(b(57)+b(58).*t1y+b(59).*t2y+b(60).*t3y) ...
          +s4p.*(b(61)+b(62).*t1y+b(63).*t2y+b(64).*t3y)) ...
      +s5.*(s1p.*(b(65)+b(66).*t1y+b(67).*t2y+b(68).*t3y) ...
          +s2p.*(b(69)+b(70).*t1y+b(71).*t2y+b(72).*t3y) ...
          +s3p.*(b(73)+b(74).*t1y+b(75).*t2y+b(76).*t3y) ...
          +s4p.*(b(77)+b(78).*t1y+b(79).*t2y+b(80).*t3y));

    s1 = fexp1(ct, pnonz(1));
    s2 = fexp1(ct, pnonz(2));
    s3 = fexp1(ct, pnonz(3));
    s4 = fexp1(ct, pnonz(4));
    s5 = fexp1(ct, pnonz(5));

    bz = s1.*((c(1)+c(2).*t1z+c(3).*t2z+c(4).*t3z) ...
          +c1p.*(c(5)+c(6).*t1z+c(7).*t2z+c(8).*t3z) ...
          +c2p.*(c(9)+c(10).*t1z+c(11).*t2z+c(12).*t3z) ...
          +c3p.*(c(13)+c(14).*t1z+c(15).*t2z+c(16).*t3z)) ...
       +s2.*((c(17)+c(18).*t1z+c(19).*t2z+c(20).*t3z) ...
          +c1p.*(c(21)+c(22).*t1z+c(23).*t2z+c(24).*t3z) ...
          +c2p.*(c(25)+c(26).*t1z+c(27).*t2z+c(28).*t3z) ...
          +c3p.*(c(29)+c(30).*t1z+c(31).*t2z+c(32).*t3z)) ...
       +s3.*((c(33)+c(34).*t1z+c(35).*t2z+c(36).*t3z) ...
          +c1p.*(c(37)+c(38).*t1z+c(39).*t2z+c(40).*t3z) ...
          +c2p.*(c(41)+c(42).*t1z+c(43).*t2z+c(44).*t3z) ...
          +c3p.*(c(45)+c(46).*t1z+c(47).*t2z+c(48).*t3z)) ...
       +s4.*((c(49)+c(50).*t1z+c(51).*t2z+c(52).*t3z) ...
          +c1p.*(c(53)+c(54).*t1z+c(55).*t2z+c(56).*t3z) ...
          +c2p.*(c(57)+c(58).*t1z+c(59).*t2z+c(60).*t3z) ...
          +c3p.*(c(61)+c(62).*t1z+c(63).*t2z+c(64).*t3z)) ...
       +s5.*((c(65)+c(66).*t1z+c(67).*t2z+c(68).*t3z) ...
          +c1p.*(c(69)+c(70).*t1z+c(71).*t2z+c(72).*t3z) ...
          +c2p.*(c(73)+c(74).*t1z+c(75).*t2z+c(76).*t3z) ...
          +c3p.*(c(77)+c(78).*t1z+c(79).*t2z+c(80).*t3z));
end

function xksi = xksi(x, y, z)
    persistent a11a12 a21a22 a41a42 a51a52 a61a62 b11b12 b21b22 c61c62 c71c72 r0 dr tnoon dteta
    if isempty(a11a12)
        a11a12 = 0.305662;
        a21a22 = -0.383593;
        a41a42 = 0.2677733;
        a51a52 = -0.097656;
        a61a62 = -0.636034;
        b11b12 = -0.359862;
        b21b22 = 0.424706;
        c61c62 = -0.126366;
        c71c72 = 0.292578;
        r0     = 1.21563;
        dr     = 7.50937;
        tnoon  = 0.3665191;
        dteta  = 0.09599309;
    end

    dr2 = dr.*dr;
    x2  = x.*x;
    y2  = y.*y;
    z2  = z.*z;
    r2  = x2+y2+z2;
    r   = sqrt(r2);
    xr  = x./r;
    yr  = y./r;
    zr  = z./r;

    if r < r0
        pr = 0.0d0;
    else
        pr = sqrt((r-r0).^2 + dr2) - dr;
    end

    f = x + pr.*(a11a12 + a21a22.*xr + a41a42.*xr.*xr + a51a52.*yr.*yr + a61a62.*zr.*zr);
    g = y + pr.*(b11b12.*yr +b21b22.*xr.*yr);
    h = z + pr.*(c61c62.*zr +c71c72.*xr.*zr);
    g2=g.*g;

    fgh    = f.^2+g2+h.^2;
    fgh32  = sqrt(fgh).^3;
    fchsg2 = f.^2+g2;

    if fchsg2 < 1.0d-5
        xksi = -1.0d0;
        return
    end

    sqfchsg2 = sqrt(fchsg2);
    alpha    = fchsg2./fgh32;
    theta    = tnoon + 0.5d0.*dteta.*(1.0d0-f./sqfchsg2);
    phi      = sin(theta).^2;

    xksi = alpha - phi;
end

function fexp = fexp(s, a)
    e = 2.718281828459d0;
    if a < 0.0d0
        fexp = sqrt(-2.0d0.*a.*e).*s.*exp(a.*s.*s);
    else
        fexp = s.*exp(a.*(s.*s-1.0d0));
    end
end


function fexp1 = fexp1(s, a)
    if a <= 0.0d0
        fexp1 = exp(a.*s.*s);
    else
        fexp1 = exp(a.*(s.*s-1.0d0));
    end
end

function tksi = tksi(xksi, xks0, dxksi)
	tdz3=2.*dxksi.^3;

    if xksi-xks0 < -dxksi 
        tksii=0.; 
    end
    
    if xksi-xks0 >= dxksi 
        tksii=1.; 
    end
    
	if xksi >= xks0-dxksi && xksi < xks0
        br3   = (xksi-xks0+dxksi)^3;
        tksii = 1.5*br3/(tdz3+br3);
	end
    
    if xksi >= xks0 && xksi < xks0+dxksi
        br3   = (xksi-xks0-dxksi)^3;
        tksii = 1.+1.5*br3/(tdz3-br3);
    end
         
    tksi=tksii;
end

function [bx, by, bz] = dipole(ps, x, y, z)
    sps = sin(ps);
    cps = cos(ps);
    
    p = x.^2;
    u = z.^2;
    v = 3.*z.*x;
    t = y.^2;
    
    q = 30574./sqrt(p+t+u).^5;
    
    bx = q.*((t+u-2..*p).*sps-v.*cps);
    by = -3..*y.*q.*(x.*sps+z.*cps);
    bz = q.*((p+t-2..*u).*cps-v.*sps);
end



