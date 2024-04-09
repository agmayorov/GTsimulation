function [ShouldBreak, Qwerty] = testBC(StepData)

    ShouldBreak = 0;
    Qwerty = StepData.R;

    if StepData.R < 0* 1.2*6370e3
        ShouldBreak = 1;
        return
    end