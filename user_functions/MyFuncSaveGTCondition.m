function a = MyFuncSaveGTCondition(GTtrack)
    a = strcmp(GTtrack.data.BC.Status, 'DefaultBC_Rmin');

%Output: 1 - if need to save
%        0 - if not need to save

%   strcmp(GTtrack.data.BC.Status, 'DefaultBC_Rmax')

%   length(GTtrack.data.Track.X)*GTtrack.data.Track.dt < 10

%   length(GTtrack.data.Track.GTmag.l2) < 10