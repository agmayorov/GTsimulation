function PlotGalaxy(inp)
%   Ver. 1, red. 1 / 31 March 2023 / A. Mayorov

    if inp == 1
        % Spiral galaxy parameter set example :
        %
        % [nb_stars = 5e4, amplitude = 0.5, radius = 1, curvature = 9, nb_arms = 2, way = -1, R0 = 3, mean_val = 2, sdn = 4, 
        % X_noise_percent = Y_noise_percent = Z_noise_percent = 0.08, a = b = 1, cmap = colormap('bone')]
        % Elliptic galaxy parameter set example #1 :
        %
        % [nb_stars = 1e5, amplitude = 3, radius = 2, curvature = 15, nb_arms = 2,
        % way = -1, R0 = 4, mean_val = 1.8, sdn = 1, a = 2, b = 1, noise = 0.15,
        % cmap = colormap('autumn')]
        % Elliptic galaxy parameter set example #2 :
        %
        % [nb_stars = 5e4, amplitude = 3, radius = 1, curvature = 360, nb_arms = 2,
        % way = -1, R0 = 4, mean_val = 3, sdn = 3, a = 2, b = 1, noise = 0.08,
        % colormap('autumn')]
        % Spiral geometric parameters

        nb_stars = 1e4;  % number of stars
        amplitude = 2;   % thickness of the galactic disk
        radius = 1;      % basis radius
        curvature = 9;   % the higher this value is, the greater the spiral is wrapped around itself
        nb_arms = 1;     % number of arms, integer
        way = -1;        % -1 / 1 : rotation way
        R0 = 3;          % radius translation parameter
        mean_val = 2;    % influence stars density at the centre
        sdn = 4;         % standard deviations number
        k = 1;           % phase at the origin
        a = 6;           % ellipsoidal X axis
        b = a;           % ellipsoidal Y axis

        % Data distribution
        R = radius*(randn(1,nb_stars)) + mean_val;
        e = find(abs(R)-mean_val < sdn*radius);
        R = R(e);
        Theta = 2*pi*randn(1,nb_stars);
        Theta = Theta(e);
        X = a*R.*cos(Theta);
        Y = b*R.*sin(Theta);
        Phi = angle(X+Y*1i);

        % Spiral shape handle function
        f = @(A,s,c,r,m,angl,p) A*exp(-(((r-mean_val).^2)/2/radius^2)/sdn^3).*sin(s*c*log(r)+m*angl+p*pi)/radius/sqrt(2*pi); 
        Z = f(amplitude,way,curvature,R+R0,nb_arms,Phi,k);

        %  Plot
        colormap('bone'); % 'bone', 'hot', 'autumn', etc.
        scatter3(X, Y, Z, abs(mean_val+sdn-R).^2, abs(Z), '*')
        axis equal; hold on
        view(-40, 15);

        xlabel('X, kpc')
        ylabel('Y, kpc')
        zlabel('Z, kpc')
    else
        [xs, ys, zs] = sphere; 
        axis equal; hold on
        view(-40, 15);
        xs = 20 * xs; ys = 20 * ys; zs = 20 * zs;
        surface(xs, ys, zs, FaceColor=[0.9 0.9 0.9], FaceAlpha=0.1, EdgeColor=[0.9 0.9 0.9])
        phi = linspace(0, 2 * pi, 100);
        fill(20*cos(phi), 20*sin(phi), [0.95 0.95 1], FaceAlpha=0.8)
        
        xlabel('X, kpc')
        ylabel('Y, kpc')
        zlabel('Z, kpc')
    end