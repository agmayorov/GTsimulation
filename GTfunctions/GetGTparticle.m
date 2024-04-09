function GTparticle_out = GetGTparticle(GTparticle_in, ifile, iev)

    if length(GTparticle_in(ifile).T) == 1
        GTparticle_out = GTparticle_in(ifile);
    else
        GTparticle_out.Nevents  = GTparticle_in(ifile).Nevents;
        GTparticle_out.Ro       = GTparticle_in(ifile).Ro(iev,:);
        GTparticle_out.Vo       = GTparticle_in(ifile).Vo(iev,:);
        GTparticle_out.T        = GTparticle_in(ifile).T(iev);
        GTparticle_out.Type     = GTparticle_in(ifile).Type(iev);
        GTparticle_out.Z        = GTparticle_in(ifile).Z(iev);
        GTparticle_out.Q        = GTparticle_in(ifile).Q(iev);
        GTparticle_out.M        = GTparticle_in(ifile).M(iev);
        GTparticle_out.PDG      = GTparticle_in(ifile).PDG(iev);
    end
    GTparticle_out.NoE = iev;
    GTparticle_out.NoF = ifile;