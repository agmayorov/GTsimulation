function GTB = GetGTBfieldParamM(MFieldFunPar, Dates, verb)

    [GetD, GetI, GetC, GetT] = deal(0);

    Date = Dates.Date;
    Year = Dates.Year;
    DoY = Dates.DoY;
    Secs = Dates.Secs;
    DTnum = Dates.DTnum;
        
    % Dip ini
    psi = 0;
    if Date == 0
        MagMom = 30100;
        dipdate = '(default)';
    else
        MagMom = GetEarthDipMagMom(Date(1:3), 'SI_nT');
        dipdate = '(date)';
    end

    % CHAOS ini
    IsIntCore = 20; 
    IsIntCrustal = 110; 
    IsExt = 0;
    ModVer = 7.13;

    % Tsyg ini
    ModCode = 96;
    if Date == 0  % quiet
        psit = 0;
        ind = [1.1200 2.0000 1.6000 -0.2000];
        tsygdate = '(default)';
    else
        psit = GetPsi([Year, DoY, Secs, DTnum]);
        ind = GetTsyganenkoInd(0, ModCode, Date);
        tsygdate = '(date)';
    end            

    if ~iscell(MFieldFunPar)
        if strcmp(MFieldFunPar, 'Dip')
            GetD = 1;
            if verb > 1
                fprintf(['           - Dip with psi = ' num2str(psi) ' rad (default)\n'])
                fprintf(['                      M   = ' num2str(MagMom) ' nT ' dipdate '\n'])
            end
        elseif strcmp(MFieldFunPar, 'IGRF')
            GetI = 1;
            IGRF = GetIGRFcoefs(Dates.DTnum);
            GTB.IGRF = IGRF;
            if verb > 1
                fprintf('           - IGRF \n')
            end
        elseif strcmp(MFieldFunPar, 'CHAOS')
            GetC = 1;
            if verb > 1
                fprintf(['           - CHAOS with Core    = ' num2str(IsIntCore) ' (default)\n'])
                fprintf(['                        Crustal = ' num2str(IsIntCrustal) ' (-/-)\n'])
                fprintf(['                        Ext     = ' num2str(IsExt) ' (-/-)\n'])
            end
        elseif strcmp(MFieldFunPar, 'Tsyg')
            GetT = 1;
            if verb > 1
                fprintf(['           - Tsyg with Mod = ' num2str(ModCode) ' (default)\n'])
                fprintf(['                       psi = ' num2str(psit) ' rad ' tsygdate '\n'])
                fprintf('                       ind = ')
                for iind = 1:numel(ind)
                    fprintf([num2str(ind(iind)) ' '])
                end
                fprintf([tsygdate '\n'])
            end
        else
            error('  TIEMF: Wrong Magnetosphere model name')
        end
    else
        for i = 1:numel(MFieldFunPar)
            if strcmp(MFieldFunPar{i},'Dip')
                GetD = 1;
                if i == numel(MFieldFunPar) || (i < numel(MFieldFunPar) && ischar(MFieldFunPar{i+1}))
                    if verb > 1
                        fprintf(['           - Dip with psi = ' num2str(psi) ' rad (default)\n'])
                        fprintf(['                      M   = ' num2str(MagMom) ' nT ' dipdate '\n'])
                    end
                    continue
                end
                [psi, MagMom] = GetDipFieldPar(MFieldFunPar{i+1}, psi, MagMom, dipdate, verb);
            elseif strcmp(MFieldFunPar(i),'IGRF')
                GetI = 1;
                if verb > 1
                    fprintf('           - IGRF \n')
                end
            elseif strcmp(MFieldFunPar(i),'CHAOS')
                GetC = 1;
                if i == numel(MFieldFunPar) || (i < numel(MFieldFunPar) && ischar(MFieldFunPar{i+1}))
                    if verb > 1
                        fprintf(['           - CHAOS with Core    = ' num2str(IsIntCore) ' (default)\n'])
                        fprintf(['                        Crustal = ' num2str(IsIntCrustal) ' (-/-)\n'])
                        fprintf(['                        Ext     = ' num2str(IsExt) ' (-/-)\n'])
                    end
                    continue
                end
                if length(MFieldFunPar{i+1}) ~= 3 && length(MFieldFunPar{i+1}) ~= 4
                    error('  TIEMF: Wrong CHAOS model number of input')
                end
                IsIntCore = MFieldFunPar{i+1}(1); 
                IsIntCrustal = MFieldFunPar{i+1}(2); 
                IsExt = MFieldFunPar{i+1}(3);
                if verb > 1
                    fprintf(['           - CHAOS with Core    = ' num2str(IsIntCore) ' (fixed by user)\n'])
                    fprintf(['                        Crustal = ' num2str(IsIntCrustal) ' (-/-)\n'])
                    fprintf(['                        Ext     = ' num2str(IsExt) ' (-/-)\n'])
                end
                if length(MFieldFunPar{i+1}) == 4
                    ModVer = MFieldFunPar{i+1}(4);
                    if verb > 1
                        fprintf(['                        ModVer     = ' num2str(ModVer) '\n'])
                    end
                end
                if IsIntCrustal > 0 && IsIntCrustal < 110
                    error('  TIEMF: IsIntCrustal < 110 - field might be wrong')
                end
            elseif strcmp(MFieldFunPar(i),'Tsyg')
                GetT = 1;
                if i == numel(MFieldFunPar) || (i < numel(MFieldFunPar) && ischar(MFieldFunPar{i+1}))
                    if verb > 1
                        fprintf(['           - Tsyg with Mod = ' num2str(ModCode) ' (default)\n'])
                        fprintf(['                       psi = ' num2str(psit) ' rad ' tsygdate '\n'])
                        fprintf('                       ind = ')
                        for iind = 1:numel(ind)
                            fprintf([num2str(ind(iind)) ' '])
                        end
                        fprintf([tsygdate '\n'])
                    end
                    continue
                end
                ModCode = MFieldFunPar{i+1}(1);
                % Refresh ind if 89                        
                if ModCode == 89
                    if numel(Date) > 1
                        ind = GetTsyganenkoInd(0, ModCode, Date);
                    end
                end
                if numel(MFieldFunPar{i+1}) == 1
                    if verb > 1
                        fprintf(['           - Tsyg with Mod = ' num2str(ModCode) ' (fixed by user)\n'])
                        fprintf(['                       psi = ' num2str(psit) ' rad ' tsygdate '\n'])
                        fprintf('                       ind = ')
                        for iind = 1:numel(ind)
                            fprintf([num2str(ind(iind)) ' '])
                        end
                        fprintf([tsygdate '\n'])
                    end
                elseif numel(MFieldFunPar{i+1}) == 3    % 89: name, psi, kp
                    if ModCode ~= 89
                        error('  TIEMF: Wrong TS 89 model input')
                    end
                    psit = MFieldFunPar{i+1}(2);
                    ind  = MFieldFunPar{i+1}(3);
                    if verb > 1
                        fprintf(['           - Tsyg with Mod = ' num2str(ModCode) ' (fixed by user)\n'])
                        fprintf(['                       psi = ' num2str(psit) ' rad (fixed by user)\n'])
                        fprintf('                       ind = ')
                        for iind = 1:numel(ind)
                            fprintf([num2str(ind(iind)) ' '])
                        end
                        fprintf('(fixed by user)\n')
                    end
                elseif numel(MFieldFunPar{i+1}) == 6    % 96: name, psi, ind(4)
                    if ModCode ~= 96
                        error('  TIEMF: Wrong TS 96 model input')
                    end
                    psit = MFieldFunPar{i+1}(2);
                    ind  = MFieldFunPar{i+1}(3:6);
                    if verb > 1
                        fprintf(['           - Tsyg with Mod = ' num2str(ModCode) ' (fixed by user)\n'])
                        fprintf(['                       psi = ' num2str(psit) ' rad (fixed by user)\n'])
                        fprintf('                       ind = ')
                        for iind = 1:numel(ind)
                            fprintf([num2str(ind(iind)) ' '])
                        end
                        fprintf('(fixed by user)\n')
                    end
                else
                    error('  TIEMF: Wrong TS model number of input')
                end                        
                if ModCode ~= 89 && ModCode ~= 96
                    error('  TIEMF: Wrong TS model name')
                end

            end
        end
    end
    if (GetI == 1 || GetC == 1 || GetT == 1) && numel(Date) == 1
        error('  TIEMF: Magnetosphere model needs Date')
    end
    if GetD + GetI + GetC + GetT == 0
        error('  TIEMF: No Magnetosphere model found')
    end

    GTB.GetD = GetD;
    GTB.GetI = GetI;
    GTB.GetC = GetC;
    GTB.GetT = GetT;

    GTB.DIP.psi = psi;
    GTB.DIP.magmom = MagMom;

    GTB.CHAOS.IsIntCore = IsIntCore;
    GTB.CHAOS.IsIntCrustal = IsIntCrustal;
    GTB.CHAOS.IsExt = IsExt;
    GTB.CHAOS.ModVer = ModVer;

    GTB.TSYG.psit = psit;
    GTB.TSYG.ind = ind;
    GTB.TSYG.ModCode = ModCode;  
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

function ind = GetTsyganenkoInd(p, ModCode, Date)
    persistent T96input date ndate
    if p == 0
        load T96input_short.mat T96input date
        ndate = [0 0 0];
    end

    if ~isequal(ndate, Date)
        ia = find(date(:,1) == Date(1) & date(:,2) == Date(2) & ...
            date(:,3) == Date(3) & date(:,4) == Date(4), 1, 'first');
        ndate = Date;
    end

    if ModCode == 89
        ind = T96input(ia, 5);
    elseif ModCode == 96
        ind = T96input(ia, 1:4);
    end
end