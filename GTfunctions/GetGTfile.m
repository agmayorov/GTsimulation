function GTfile = GetGTfile(GTparam)

    GTfile = '';

    if ~isempty(GTparam.IOinfo.SaveFile)
        SaveDir  = GTparam.IOinfo.SaveDir;
        SaveFile = GTparam.IOinfo.SaveFile;

        Slash = '';
        if ~isempty(SaveDir)
            if ~endsWith(SaveDir, '/')
                Slash = '/';
            end
        end

        Mat = '';
        if ~endsWith(SaveFile, '.mat')
            Mat = '.mat';
        end
            
        GTfile.dir.param    = [SaveDir Slash 'GTparam/'];
        GTfile.dir.particle = [SaveDir Slash 'GTparticle/'];
        GTfile.dir.track	= [SaveDir Slash 'GTtrack/'];
        
        GTfile.file.param    = [SaveDir Slash 'GTparam/GTparam_' SaveFile Mat];
        GTfile.file.particle = [SaveDir Slash 'GTparticle/GTparticle_' SaveFile Mat];
        for ifile = 1:GTparam.IOinfo.Nfiles
            if GTparam.IOinfo.Nfiles == 1
            	GTfile.file.track(ifile).fname = [SaveDir Slash 'GTtrack/GTtrack_' SaveFile Mat];
            else
            	GTfile.file.track(ifile).fname = [SaveDir Slash 'GTtrack/GTtrack_' SaveFile '_' num2str(ifile) Mat];
            end
            if strcmp(GTparam.IOinfo.LoadGTtrack, 'on')
                mustBeFile(GTfile.file.track(ifile).fname);
            end
        end
    end
