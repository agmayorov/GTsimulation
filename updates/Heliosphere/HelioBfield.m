classdef HelioBfield < BfieldAbs
    
    properties
        mag
        cir
        pol
        T0
        NParam = struct("num", 1024, "log_kmin", 0, "log_kmax", 7)
        use_noise = 0
        
        % The radius of magnetic field surface
        rs = 0.0232523
        % Angular speed of the Sun in s^-1 (25.4 days)
        omega = 2*pi./2160000;
    end
    
    methods
        function self = HelioBfield(HParam)
            arguments
                HParam.mag = 2.09
                HParam.cir = 0
                HParam.pol = -1
                HParam.T0 = 0
                HParam.NParam = struct("num", {}, "log_kmin", {}, "log_kmax", {})
            end
            self.ModelName = "Helio";
            self.Region = "H";
            self.mag = HParam.mag;
            self.cir = HParam.cir;
            self.pol = HParam.pol;
            self.T0 = HParam.T0;
            if ~isempty(HParam.NParam)
                self.set_noise(HParam.NParam);
            end
        end
        
        function [Bx, By, Bz] = GetBfield(self, x, y, z, t)
            % Magetic field magnitue at earth surafce multiplied at 1 au
            A0 = self.mag.*self.pol;    
            % 11 years to seconds
            ele = 347133600;          

            [r, R, theta, phi] = HelioBfield.Cart2Sphere(x, y, z);
            t = t + self.T0;

            alpha = HelioBfield.calc_tilt(t);
            dalpha = sign(HelioBfield.calc_tilt(t+1e-6) - HelioBfield.calc_tilt(t-1e-6));
            alpha = alpha - pi./ele.*(r-self.rs)./self.v_wind(theta).*dalpha;

            theta0 = pi./2 - atan(-tan(alpha).*sin(phi + self.omega.*(r-self.rs)./self.v_wind(theta) - self.omega.*t));
            %The step function for decression of el.mag. field becous of HCS 
            H = HelioBfield.StepFunc(theta, theta0, r);   
            %Magnetic sperical regular heliosphere components
            Br = (A0./r.^2).*H.*(r>=self.rs); 
            Bphi = -(A0./r.^2).*(((r-self.rs).*self.omega)./self.v_wind(theta)).*sin(theta).*H.*(r>=self.rs);

            if self.cir == 1
                % CIR function
            end

            [Bx, By, Bz] = HelioBfield.Sphere2Cart(Br, 0, Bphi, theta, phi);

            if self.use_noise == 1
                coeff2d = 1.4; 
                coeffslab = coeff2d./2;
                [Bx_slab, By_slab, Bz_slab] = self.GetSlabfield(x, y, z);
                [Bx_2d, By_2d, Bz_2d] = self.Get2Dfield(x, y, z);
                Bx = Bx + self.mag.*coeff2d.*Bx_2d + self.mag.*coeffslab.*Bx_slab;
                By = By + self.mag.*coeff2d.*By_2d + self.mag.*coeffslab.*By_slab;
                Bz = Bz + self.mag.*coeff2d.*Bz_2d + self.mag.*coeffslab.*Bz_slab;
            end
        end

        function [Bx, By, Bz] = GetSlabfield(self, x, y, z)
            q = 5/3;                            % High wavenumbers spectrum
            p = 0;                              % Low wavenumbers specreum
            gamma = 3;                          % Turbulence energy power law

            [r, R, theta, phi] = HelioBfield.Cart2Sphere(x, y, z);

            if r<self.rs
                Bx = 0;
                By = 0;
                Bz = 0;
                return;
            end

            cospsi = 1./sqrt(1+(r.*sin(theta)./self.a(theta)).^2);
            sinpsi = (r.*sin(theta)./self.a(theta))./sqrt(1+(r.*sin(theta)./self.a(theta)).^2);

            num = self.NParam.num;
            log_kmin = self.NParam.log_kmin;
            log_kmax = self.NParam.log_kmax;
            k = reshape(logspace(log_kmin, log_kmax, num), num, 1);
            dk = k .* (10.^((log_kmax - log_kmin)./(num-1)) - 1);
        
            lambda = 0.08.*(r./self.rs).^(0.8).*self.rs;
            numer = dk.*k.^p;

            % Radial spectrum

            A_rad = self.NParam.A_rad;
            B_rad = A_rad.*(r).^(-gamma./2);
            brk_rad = lambda.*k./sqrt(self.a(theta).*r);
            denom_rad = (1+brk_rad.^(p+q));
            
            spectrum_rad = sqrt(numer./denom_rad);
            deltaB_rad = 2.*B_rad.*spectrum_rad.*cospsi.*r.*sqrt(r.*self.a(theta));

            % Azimuthal spectrum

            A_az = self.NParam.A_az;
            B_az = A_az.*(r).^(-gamma./2);
            brk_az = lambda.*k./r;
            denom_az = (1+brk_az.^(p+q));
            spectrum_az = sqrt(numer./denom_az);
        
            deltaB_az = B_az.*spectrum_az;
            dpsectrum_az = 0.08.*(p+q).*numer.*brk_az.^(p+q-1)./...
                           (2*spectrum_az.*(denom_az.^2)).* ... 
                           (0.8.*k./(r.*(r./self.rs).^(0.2)) - brk_az./(0.08.*r));

            % Radial polarization and phase

            alpha_rad = self.NParam.alpha_rad;
            n = round(sin(alpha_rad).*k);
            alpha_rad = real(asin(n./k).*(cos(alpha_rad)>0) + (pi - asin(n./k)).*(cos(alpha_rad)<0));

            delta_rad = self.NParam.delta_rad;
            phase_rad = k.*sqrt(r./self.a(theta)) + delta_rad;

            % Azimuthal polarization and phase

            alpha_az = self.NParam.alpha_az;
            n = round(sin(alpha_az).*k);
            alpha_az = real(asin(n./k).*(cos(alpha_az)>0) + (pi - asin(n./k)).*(cos(alpha_az)<0));

    
            delta_az = self.NParam.delta_az;
            phase_az = k.*phi + delta_az;

            % Radial field
            Br_rad = 0;
            
            Btheta_rad =-deltaB_rad.*cos(phase_rad).*csc(theta).*sin(alpha_rad)./ ... 
                        (2.*r.*sqrt(r./self.a(theta)));
        
            Bphi_rad = deltaB_rad./r.*cos(alpha_rad).*cospsi.*cos(phase_rad)./ ... 
                        (2.*r.*sqrt(r./self.a(theta)));
        
        
            % Azimuthal field
        
            Br_az = -deltaB_az.*sinpsi.*cos(alpha_az).*cos(phase_az);
            Btheta_az = deltaB_az.*sinpsi.*sin(alpha_az).*cos(phase_az);
            Bphi_az = (deltaB_az.*(1-gamma./2).*cos(alpha_az).*sin(theta).* ... 
                                                 sinpsi.*sin(phase_az) -  ... 
                       deltaB_az.*sinpsi.*sin(phase_az).*(cos(theta).*sin(alpha_az) - ...
                                                          sin(theta).*cos(alpha_az)) + ...
                       B_az./r .*dpsectrum_az.*cos(alpha_az).*sin(theta).*sin(phase_az).*sinpsi)./k;
        
            % Total field
            Br = sum(Br_az+Br_rad);
            Btheta = sum(Btheta_az+Btheta_rad);
            Bphi = sum(Bphi_az+Bphi_rad);

            [Bx, By, Bz] = HelioBfield.Sphere2Cart(Br, Btheta, Bphi, theta, phi);
        end

        function [Bx, By, Bz] = Get2Dfield(self, x, y, z)
            q = 8/3;                            % High wavenumbers spectrum
            p = 0;                              % Low wavenumbers specreum
            gamma = 3;                          % Turbulence energy power law

            [r, R, theta, phi] = HelioBfield.Cart2Sphere(x, y, z);

            if r<self.rs
                Bx = 0;
                By = 0;
                Bz = 0;
                return;
            end

            cospsi = 1./sqrt(1+(r.*sin(theta)./self.a(theta)).^2);
            sinpsi = (r.*sin(theta)./self.a(theta))./sqrt(1+(r.*sin(theta)./self.a(theta)).^2);


            num = self.NParam.num;
            log_kmin = self.NParam.log_kmin;
            log_kmax = self.NParam.log_kmax;
            k = reshape(logspace(log_kmin, log_kmax, num), num, 1);
            dk = k .* (10.^((log_kmax - log_kmin)./(num-1)) - 1);

            A = self.NParam.A_2D;
            lambda = 0.04.*(r./self.rs).^(0.8).*self.rs;
            B = A.*(r).^(-gamma./2);
            brk = lambda.*k./r;
            denom = (1+brk.^(p+q));
            numer = dk.*k.^(p+1);
            spectrum = sqrt(2.*pi.*numer)./sqrt(denom);
            deltaB = B.*spectrum;
        
            dspectrum = - (0.04.*sqrt(pi./2).*(p+q).*numer.*(brk).^(p+q-1))./ ... 
                          (spectrum./sqrt(2*pi).*denom.^2).* ... 
                          (0.8.*k./(r.*(r/self.rs).^(0.2)) - brk./(0.04.*r));

            alpha = self.NParam.alpha_2D;
            n = round(sin(alpha).*k);
            alpha = real(asin(n./k).*(cos(alpha)>0) + (pi - asin(n./k)).*(cos(alpha)<0));
        
            
            delta = self.NParam.delta_2D;
            phase = k.*(r./self.a(theta) + phi + theta.*cos(alpha)) + delta;

            Br = -deltaB./r.*sinpsi.*(cos(alpha).*cos(phase) + ... 
                              (cot(theta).*sin(phase))./k);
    
            Btheta = deltaB.*(sinpsi.*cos(phase)./self.a(theta) - ... 
                              gamma.*sinpsi.*sin(phase)./(2.*r.*k) + ...
                              csc(theta)./r.*(cospsi.*cos(phase) + ... 
                                              sin(theta).*sinpsi.*sin(phase)./k)) + ...
                     B.*sinpsi.*sin(phase).*dspectrum./k;
        
            Bphi = -deltaB./r.*cos(alpha).*cospsi.*cos(phase);

            Br = sum(Br);
            Btheta = sum(Btheta);
            Bphi = sum(Bphi);

            [Bx, By, Bz] = HelioBfield.Sphere2Cart(Br, Btheta, Bphi, theta, phi);
        end
    end

    methods (Access = private)
        function set_noise(self, NParam)
            self.use_noise = 1;
            if isfield(NParam, "num")
                self.NParam.num = NParam.num;
            end

            if isfield(NParam, "log_kmin")
                self.NParam.log_kmin = NParam.log_kmin;
            end

            if isfield(NParam, "log_kmax")
                self.NParam.log_kmax = NParam.log_kmax;
            end 

            num = self.NParam.num;

            self.NParam.A_2D = randn(num, 1)./(130);
            self.NParam.alpha_2D = rand(num, 1).*2.*pi;
            self.NParam.delta_2D = rand(num, 1)*.2.*pi;

            self.NParam.A_rad = randn(num, 1)./1.5;
            self.NParam.alpha_rad = rand(num, 1).*2.*pi;
            self.NParam.delta_rad = rand(num, 1)*.2.*pi;

            self.NParam.A_az = randn(num, 1)./4.5;
            self.NParam.alpha_az = rand(num, 1).*2.*pi;
            self.NParam.delta_az = rand(num, 1)*.2.*pi;
        end
        
        function v = v_wind(~, theta)
            v = (300 + 475 .* (1-sin(theta).^8)).*6.7e-9;
        end

        function frac = a(self, theta)
            frac = self.v_wind(theta)/self.omega;
        end
    end

    methods (Static)
        function alpha = calc_tilt(t)
            % Input: t in seconds
            % Output: tilt angle at source surface in radians
        
            a0 = 0.7502;
            a1 = 0.02332;  
            b1 = -0.01626;  
            a2 = -0.3268; 
            b2 = 0.2016; 
            a3 = -0.02814;  
            b3 = 0.0005215;  
            a4 = -0.08341;  
            b4 = -0.04852;  
            w = 9.318e-09; 
            
            alpha = a0 + a1.*cos(t.*w) + b1.*sin(t.*w) + a2.*cos(2.*t.*w) + b2.*sin(2.*t.*w) + a3.*cos(3.*t.*w) + b3.*sin(3.*t.*w) + a4.*cos(4.*t.*w) + b4.*sin(4.*t.*w);
        end

        function H = StepFunc(theta, theta0, r)
                L0 = 0.0002;
                dt = r.*(theta - theta0)./L0;                                          
                H = -tanh(dt);
        end
    
        function [r, R, theta, phi] = Cart2Sphere(x, y, z)
            r = sqrt(x.^2 + y.^2 + z.^2 + 1e-12);
            R = sqrt(x.^2 + y.^2+ 1e-12);
            theta = acos(z./r);
            phi = acos(x./R).*(y>=0)+(2*pi - acos(x./R)).*(y<0);
        end
    
        function [Bx, By, Bz] = Sphere2Cart(Br, Btheta, Bphi, theta, phi)
            Bx = Br.*sin(theta).*cos(phi) +Btheta.*cos(theta).*cos(phi) - Bphi.*sin(phi);
            By = Br.*sin(theta).*sin(phi) + Btheta.*cos(theta).*sin(phi) + Bphi.*cos(phi);             
            Bz = Br.*cos(theta) - Btheta.*sin(theta);
        end
    end
end

