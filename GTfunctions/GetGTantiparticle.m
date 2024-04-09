function GTantiparticle = GetGTantiparticle(GTparticle)

    GTantiparticle = GTparticle;

    GTantiparticle.Z    = - GTantiparticle.Z;
    GTantiparticle.Q    = - GTantiparticle.Q;
    GTantiparticle.PDG  = - GTantiparticle.PDG;
    GTantiparticle.Rig  = - GTantiparticle.Rig;

    if strcmp(GTparticle.Type{:}(1), 'a')
        GTantiparticle.Type = {GTparticle.Type{:}(2:end)};
    else
        GTantiparticle.Type = {['a' GTparticle.Type{:}]};
    end