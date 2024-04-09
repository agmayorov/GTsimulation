function Date = DateTimeRedefine(Date, dt)
    t = datetime(Date(1), Date(2), Date(3), Date(4), Date(5), Date(6) + dt, ...
                'TimeZone', 'UTC', 'Format', 'dd-MMM-uuuu HH:mm:ss.SSS');
    Date(1) = t.Year;
    Date(2) = t.Month;
    Date(3) = t.Day;
    Date(4) = t.Hour;
    Date(5) = t.Minute;
    Date(6) = t.Second;
end