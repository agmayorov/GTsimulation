function GTtrack = RunGetTrajectoryInEMField(GTparam, GTparticle)

    GTtrack = TNode();
    for ifile = 1:GTparam.IOinfo.Nfiles
        GTparam.No.File = ifile;
        if GTparam.Verbose
            tic
        end
        n = 0;
        for iev = 1 : GTparticle(ifile).Nevents
            GTparam.No.Event = iev;
            GTtrack(iev - n) = GetTrajectoryInEMField(GTparam, ...
                                                  GetGTparticle(GTparticle, ifile, iev), ...
                                                  GetGTtrack(GTparam, ifile, iev));
            if GTparam.SaveMode.UseFunction
                if ~GTparam.SaveMode.UserSaveGTCondition(GTtrack(iev - n))
                    GTtrack(iev - n) = [];
                    n = n + 1;
                end
            end
        end        
		if GTparam.Verbose > 1
            fprintf('\n') 
        end
        if GTparam.Verbose
            fprintf(['   <> <> <> <> <> ' num2str(toc) ' <> <> <> <> <> \n'])
        end
        if GTparam.Verbose > 1
            fprintf('\n') 
        end
        if ~isempty(GTparam.GTfile)
            if GTparam.Verbose
                fprintf('   Save GTtrack to file                    ...')
            end    
            save(GTparam.GTfile.file.track(ifile).fname, 'GTtrack')
            if GTparam.Verbose
                fprintf('   done \n')
            end
        end
    end
