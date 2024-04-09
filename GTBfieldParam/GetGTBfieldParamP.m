function GTB = GetGTBfieldParamP(MFieldFunPar)

    GetP = 0;

    psi = 0;
    if Date == 0
        MagMom = 50e16;
        dipdate = '(default)';
    else
        MagMom = GetEarthDipMagMom(Date(1:3), 'SI_nT');
        dipdate = '(date)';
    end

    if ~iscell(MFieldFunPar)
        if strcmp(MFieldFunPar, 'Dip')
            GetP = 1;
            if verb > 1
                fprintf(['           - Dip with psi = ' num2str(psi) ' rad (default)\n'])
                fprintf(['                      M   = ' num2str(MagMom) ' nT (-/-)\n'])
            end
        else
            error('  TIEMF: Wrong Pulsar MF model name')
        end
    else
        for i = 1:numel(MFieldFunPar)
            if strcmp(MFieldFunPar{i},'Dip')
                GetP = 1;
                if i == numel(MFieldFunPar)
                    if verb > 1
                        fprintf(['           - Dip with psi = ' num2str(psi) ' rad (default)\n'])
                        fprintf(['                      M   = ' num2str(MagMom) ' nT (-/-)\n'])
                    end
                    break
                end
                [psi, MagMom] = GetDipFieldPar(MFieldFunPar{i+1}, psi, MagMom, dipdate, verb);
            end
        end
    end 

    if sum(GetP) == 0
        error('  TIEMF: No ISMF model found')
    end

    GTB.GetP = GetP;

    GTB.DIP.psi = psi;
    GTB.DIP.magmom = MagMom;
end

function [psi, MagMom] = GetDipFieldPar(MFieldFunPar, psi, MagMom, dipdate, verb)
    if ischar(MFieldFunPar)
        if verb > 1
            fprintf(['           - Dip with psi = ' num2str(psi) ' rad (default)\n'])
            fprintf(['                      M   = ' num2str(MagMom) ' nT (-/-)\n'])
        end
        return
    end
    if numel(MFieldFunPar) == 2
        if MFieldFunPar(1) < 2*pi
            psi = MFieldFunPar(1);
            MagMom = MFieldFunPar(2);
        else
            psi = MFieldFunPar(2);
            MagMom = MFieldFunPar(1);
        end
        if verb > 1
            fprintf(['           - Dip with psi = ' num2str(psi) ' rad (fixed by user)\n'])
            fprintf(['                      M   = ' num2str(MagMom) ' nT (-/-)\n'])
        end
    elseif numel(MFieldFunPar) == 1
        if MFieldFunPar < 2*pi
            psi = MFieldFunPar;
            if verb > 1
                fprintf(['           - Dip with psi = ' num2str(psi) ' rad (fixed by user)\n'])
                fprintf(['                      M   = ' num2str(MagMom) ' nT ' dipdate '\n'])
            end
        else
            MagMom = MFieldFunPar;
            if verb > 1
                fprintf(['           - Dip with psi = ' num2str(psi) ' rad (default)\n'])
                fprintf(['                      M   = ' num2str(MagMom) ' nT (fixed by user)\n'])
            end
        end
    else
        error('  TIEMF: Wrong Pulsar MF model number of input')
    end  
end