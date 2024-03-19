%% Gauss field test
GaussField = Gauss('Model', 'IGRF', 'Ver', 13, 'Type', 'core', 'Date', [2006 06 15]);
[BxG, ByG, BzG] = GaussField.GetBfield(1000, 1000, 1000);

%% Tsyganenko field test
TsyganenkoField = Tsyganenko("ps", 0, "iopt", [1.1200 2.0000 1.6000 -0.2000], "Year", 2015, "DoY", 212, "Secs", 43200, "ModCode", 96);
[BxT, ByT, BzT] = TsyganenkoField.GetBfield(10, 10, 10);