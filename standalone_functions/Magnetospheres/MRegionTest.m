%% Gauss field test
GaussField = Gauss('Model', 'IGRF', 'Ver', 13, 'Type', 'core', 'Date', [2006 06 15]);
[BxG, ByG, BzG] = GaussField.GetBfield(1000, 1000, 1000);

%% Tsyganenko field test
TsyganenkoField = Tsyganenko("Date", [2008 1 1 12 0 0], "ModCode", 96);
[BxT, ByT, BzT] = TsyganenkoField.GetBfield(10, 10, 10);