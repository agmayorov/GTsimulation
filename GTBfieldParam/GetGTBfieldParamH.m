function GTB = GetGTBfieldParamH(MFieldFunPar)

    GetH = 0;
    pol = -1;
    cir = 0;
    noise = 0;
    T0 = 2008 * 365.25 * 24 * 3600;
    
    if ~iscell(MFieldFunPar)
        if strcmp(MFieldFunPar, 'Helio')
            GetH = 1;
%             if verb > 1
%                 fprintf(['           - Helio with pol   = ' num2str(pol) ' (default)\n'])
%                 fprintf(['                        cir   = ' num2str(cir) ' (-/-)\n'])
%                 fprintf(['                        noise = ' num2str(noise) ' (-/-)\n'])
%             end
        else
            error('  TIEMF: Wrong ISMF model name')
        end
    else
        for i = 1:numel(MFieldFunPar)
            if strcmp(MFieldFunPar{i}, 'Helio')
                GetH = 1;
            end
            if strcmp(MFieldFunPar{i}, "pol")
                pol   = MFieldFunPar{i+1};
%                 fprintf(['           - Helio with pol   = ' num2str(pol) ' (fixed by user)\n'])
            end
            if strcmp(MFieldFunPar{i}, "cir")
                cir   = MFieldFunPar{i+1};
%                 fprintf(['                        cir   = ' num2str(cir) ' (-/-)\n'])
            end
            if strcmp(MFieldFunPar{i}, "noise")
                noise = MFieldFunPar{i+1};
%                 if verb > 1
%                     fprintf(['                        noise = ' num2str(~isempty(noise)) ' (-/-)\n'])
%                 end
                if ~isempty(noise) == 1
                    num = 2048;
                    log_kmin = 0;
                    log_kmax = 7;
                    for param_index = 1:length(noise)
                        p = noise{param_index};
                        if strcmp(p,"number")
                            num = noise{param_index+1};
                        elseif strcmp(p,"log_kmin")
                            log_kmin = noise{param_index+1};
                        elseif strcmp(p, "log_kmax")
                            log_kmax = noise{param_index+1};
                        end
                        A_2D = randn(num, 1)./(130);
                        alpha_2D = rand(num, 1).*2.*pi;
                        delta_2D = rand(num, 1)*.2.*pi;

                        A_rad = randn(num, 1)./1.5;
                        alpha_rad = rand(num, 1).*2.*pi;
                        delta_rad = rand(num, 1)*.2.*pi;

                        A_az = randn(num, 1)./4.5;
                        alpha_az = rand(num, 1).*2.*pi;
                        delta_az = rand(num, 1)*.2.*pi;

                        noise = {"number", num, "log_kmin", log_kmin, "log_kmax", log_kmax, ...
                                "2D", {"A", A_2D, "alpha", alpha_2D, "delta", delta_2D}, ...
                                "Slab", {"A_rad", A_rad, "alpha_rad", alpha_rad, "delta_rad", delta_rad, ...
                                         "A_az", A_az, "alpha_az" alpha_az, "delta_az", delta_az}};
                    end                    
%                     if verb > 1
%                         fprintf(['                        grid_size = ' num2str(noise{1}) ' (-/-)\n'])
%                         fprintf(['                        cell_num = ' num2str(noise{2}) ' (-/-)\n'])
%                         fprintf('                         - Generating noise ... \n')
%                         fprintf('                           Progress: ')
%                     end
                      
%                     if verb > 1
%                         fprintf('\n')
%                     end
                end
            end
        end
    end  
    if sum(GetH) == 0
        error('  TIEMF: No IPMF model found')
    end

    GTB.GetH = GetH;

    GTB.T0 = T0;
    GTB.pol = pol;
    GTB.cir = cir;
    GTB.noise = noise;
