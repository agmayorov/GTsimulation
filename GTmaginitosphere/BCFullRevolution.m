function [ShouldBreakUserFunction, UserParam] = BCFullRevolution(StepData)
    ShouldBreakUserFunction = 0;
    UserParam = StepData.UserParam;
    
    [a, b, c] = geo2mag_eccentric_gh(StepData.XYZCur(1), StepData.XYZCur(2), StepData.XYZCur(3), 1, StepData.IGRF.g, StepData.IGRF.h);
    XYZ_in_MAG_eccentric = [a, b, c];

    lon = atan(XYZ_in_MAG_eccentric(2) / XYZ_in_MAG_eccentric(1));
    lon_tolerance = deg2rad(60);

    if StepData.istep == 1
        UserParam.full_revolutions = 0;
        UserParam.lon_prev = lon;
        UserParam.lon_total = 0;
    end    

    lon_diff = abs(lon - UserParam.lon_prev);
    lon_diff(lon_diff > pi/2) = pi - lon_diff;

    if lon_diff > lon_tolerance
        UserParam.lon_total = UserParam.lon_total + lon_diff;
        UserParam.lon_prev = lon;
    end
    
    UserParam.lon_total = abs(UserParam.lon_total);

    is_full_revolution = abs(UserParam.lon_total) > 2*pi;
    
    if is_full_revolution
        UserParam.full_revolutions = UserParam.full_revolutions + 1;
        UserParam.lon_total = rem(UserParam.lon_total, 2*pi);
    end

    if UserParam.full_revolutions == 1
        ShouldBreakUserFunction = 1;
    end