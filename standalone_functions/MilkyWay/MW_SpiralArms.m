function theta_arm = MW_SpiralArms(R)

alpha = [3.33783 4.36017 5.36672 4.72076];
Rmin  = [2.03805 3.31587 3.94189 3.16184];

theta_arm = alpha .* log10(R./Rmin) + (1:4)*pi/2-pi/6;
%pitch_arm = atan(1./alpha);