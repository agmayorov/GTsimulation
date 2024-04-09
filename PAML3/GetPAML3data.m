function PAML3 = GetPAML3data(d, ev)

    % LoadL3vars
    
    load(['PAML3/data/Exposure_' num2str(d) '.mat'], 'Date', 'lat', 'lon', 'alt')
    load(['PAML3/data/PitchAngles_' num2str(d) '.mat'], 'Sij')
    load(['PAML3/data/NTrack_' num2str(d) '.mat'], 'Event')
    load(['PAML3/data/Tracker_' num2str(d) '.mat'], 'Rig')

    PAML3.Date = Date(Event(ev), :);

    PAML3.lat = lat(Event(ev), :);
    PAML3.lon = lon(Event(ev), :);
    PAML3.alt = alt(Event(ev), :);

    PAML3.Sij = Sij(ev, :);

    PAML3.Rig = Rig(ev, :);