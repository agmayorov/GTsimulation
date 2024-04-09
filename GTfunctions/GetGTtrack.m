function GTtrack_out = GetGTtrack(GTparam, ifile, iev)

    GTtrack_out = [];

    if strcmp(GTparam.IOinfo.LoadGTtrack, 'on')
        load(GTparam.GTfile.file.track(ifile).fname, 'GTtrack');
        GTtrack_out = GTtrack(iev);
    end