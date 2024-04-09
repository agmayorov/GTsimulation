function GTB = GetGTBfieldParamG(MFieldFunPar, verb)

    GetG = 0;
    if ~iscell(MFieldFunPar)
        if strcmp(MFieldFunPar, 'J12')
            GetG = 1;
            if verb > 1
                fprintf('           - J12 \n')
            end
        else
            error('  TIEMF: Wrong ISMF model name')
        end
    else
        for i = 1:numel(MFieldFunPar)
            if strcmp(MFieldFunPar{i},'J12')
                GetG = 1;
                if verb > 1
                    fprintf('           - J12 \n')
                end
            end
            % GetJ12FieldPar
        end
    end            
    if sum(GetG) == 0
        error('  TIEMF: No ISMF model found')
    end

    GTB.GetG = GetG;
