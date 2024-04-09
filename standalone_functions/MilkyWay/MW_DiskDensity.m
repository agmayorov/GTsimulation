function F_d = MW_DiskDensity(R, Z)
    
r_s = 8;
n_d = ;
r_o = 1.12804;
r_h = 6.43578;
h_i = 7.30415;

f_d = n_d * exp(-(R - r_s)/r_o) * (1 - exp( (-R/r_h)^h_i ));