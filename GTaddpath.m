function GTaddpath

    addpath(genpath('GTfunctions'))
    addpath('GTBfieldParam')
    addpath('GTmaginitosphere')
    addpath('user_functions')
    addpath('PAML3')
    addpath(genpath('standalone_functions'))
    if ispc
        addpath(genpath('D:/lustre/mFunctions'))
	else
        addpath(genpath('/lustre/mFunctions'))
    end