function RandSeed = GetRandomSeed
%   Ver. 1, red. 1 / 31 March 2023 / A. Mayorov

    if ispc
        [~, RandSeed] = system('echo %random%');
    else
        [~, RandSeed] = system('echo $RANDOM'); 
    end
    
    RandSeed = str2double(RandSeed);