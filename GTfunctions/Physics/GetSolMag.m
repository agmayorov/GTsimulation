function SolMag = GetSolMag(Date)
    [DATE(1), DATE(2), ~, DATE(3)] = DateRedefine(Date);

    persistent f107Daily f107Average magneticIndex dayOfYear date ndate doy Fav MI F

    if isempty(f107Average)
        load atmosnrlmsise00_input.mat f107Daily f107Average magneticIndex dayOfYear date
        ndate = [0 0 0];
    end
    if ~isequal(ndate, DATE)
        ia = find(date(:,1) == DATE(1) & dayOfYear == DATE(2), 1, 'first');
        % f107Daily, f107Average, Magnetic Index during a day ia
        doy = dayOfYear(ia);
        Fav = f107Average(ia);
        MI = magneticIndex(ia,:);
        F = NaN;
        while isnan(F)
            F = f107Daily(ia);
            ia = ia + 1;
        end
        ndate = DATE;
    end

    % Day of year
    % UT seconds in day
    % f10.7
    % f10.7av
    % Magnetic index
    SolMag = [doy DATE(3) F Fav MI(1)];
end