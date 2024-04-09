function [ShouldBreak, TimesCrossedMagEquator] = MyBCFunction(StepData)
    
    if isempty(StepData.UserParam)
        TimesCrossedMagEquator = 0;
    else
        TimesCrossedMagEquator = StepData.UserParam;
    end

    XYZInMag     = geo2mag(StepData.XYZCur);
    XYZInMagPrev = geo2mag(StepData.XYZPrev);
    
    DidZSignChange = XYZInMag(3) * XYZInMagPrev(3) <= 0;
    
    if DidZSignChange
        TimesCrossedMagEquator = TimesCrossedMagEquator + 1;
    end
    
    ShouldBreak = TimesCrossedMagEquator > 2;
end

function a = geo2mag(b)

    if size(b, 2) == 3
        b = b';
    end
    a = [ 0.339067758413505 -0.919633920274268 -0.198258689306225 ; ...
            0.938257039240758 0.345938908356903 0 ; ...
            0.068589929661063 -0.186019809236783 0.980148994857721] * b;
end