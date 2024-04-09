function GTparam_upd = RefreshGTparam(GTargs)
    arguments
        GTargs.GTparam      struct
        GTargs.Ro           cell
        GTargs.Vo           cell
        GTargs.Particle     cell
        GTargs.Dates        cell
        GTargs.EMFF         cell
        GTargs.Steps        cell
        GTargs.Region       char
        GTargs.Medium       cell
        GTargs.InteractEM   cell
        GTargs.InteractNUC  cell
        GTargs.SaveMode     cell
        GTargs.BC           cell
        GTargs.GTmag        cell
        GTargs.Nevents      double
        GTargs.IOinfo       cell
        GTargs.Verbose      double
    end
    
    if ~isfield(GTargs, 'GTparam')
        error('GTparam not defined')
    end
    
    GTparam = GTargs.GTparam;
    
    Ro = GTparam.Ro.input;
    if isfield(GTargs, 'Ro')
        Ro = gtupdate(Ro, GTargs.Ro);
    end

    Vo = GTparam.Vo.input;
    if isfield(GTargs, 'Vo')
        Vo = gtupdate(Vo, GTargs.Vo);
    end

    Particle = GTparam.Particle.input;
    if isfield(GTargs, 'Particle')
        Particle = gtupdate(Particle, GTargs.Particle);
    end

    Dates = GTparam.Dates;
    if isfield(GTargs, 'Dates')
        Dates = gtupdate(Dates, GTargs.Dates);
    end
    
    EMFF = GTparam.EMFF.input;
    if isfield(GTargs, 'EMFF')
        EMFF = gtupdate(EMFF, GTargs.EMFF);
    end
    
    Steps = GTparam.Steps.input;
    if isfield(GTargs, 'Steps')
        Steps = gtupdate(Steps, GTargs.Steps);
    end
    
    if ~isfield(GTargs, 'Region'), Region = GTparam.Region.Name;
    else,                          Region = GTargs.Region;
    end
    
    Medium = GTparam.Medium.input;
    if isfield(GTargs, 'Medium')
        Medium = gtupdate(Medium, GTargs.Medium);
    end

    InteractEM = GTparam.InteractEM.input;
    if isfield(GTargs, 'InteractEM')
        InteractEM = gtupdate(InteractEM, GTargs.InteractEM);
    end
    
    InteractNUC = GTparam.InteractNUC.input;
    if isfield(GTargs, 'InteractNUC')
        InteractNUC = gtupdate(InteractNUC, GTargs.InteractNUC);
    end
    
    SaveMode = GTparam.SaveMode.input;
    if isfield(GTargs, 'SaveMode')
        SaveMode = gtupdate(SaveMode, GTargs.SaveMode);
    end
    
    BC= GTparam.BC.input;
    if isfield(GTargs, 'BC')
        BC = gtupdate(BC, GTargs.BC);
    end
    
    GTmag = GTparam.GTmag.input;
    if isfield(GTargs, 'GTmag')
        GTmag = gtupdate(GTmag, GTargs.GTmag);
    end
    
    if ~isfield(GTargs, 'Nevents'), Nevents = GTparam.Nevents;
    else,                           Nevents = GTargs.Nevents;
    end
    
    IOinfo = GTparam.IOinfo.input;
    if isfield(GTargs, 'IOinfo')
        IOinfo = gtupdate(IOinfo, GTargs.IOinfo);
    end
    
    if ~isfield(GTargs, 'Verbose'), Verbose = GTparam.Verbose;
    else,                           Verbose = GTargs.Verbose;
    end

    GTparam_upd = SetGTparam(Ro, Vo, Particle, Dates.Date, EMFF, Region, Steps, Medium, InteractEM, InteractNUC, ...
                                                                SaveMode, BC, GTmag, Nevents, IOinfo, Verbose);

end

function Po = gtupdate(Po, Po_upd)
    for i = 1:2:length(Po_upd)
        if ~ischar(Po_upd{i})
            error('Wrong name of input parameter')
        end

        n = find(strcmp(Po, Po_upd{i}));
        default = 0;
        if strcmp(Po_upd{i+1}, 'default') || strcmp(Po_upd{i+1}, 'def')
            default = 1;
        end

        if n
            if ~default
                Po{n+1} = Po_upd{i+1};
            else
                Po(n:n+1) = [];
            end
        else
            if ~default
                Po{end+1} = Po_upd{i}; %#ok<*AGROW> 
                Po{end+1} = Po_upd{i+1};
            end
        end
    end
end
