function ChemCompRe = ChemCompRedefine(ChemComp)
%    (He, O, N2, O2, Ar, H, N) (O+ N+ H+ He+ O2+ NO+) (H2)
    w_He = ChemComp(1) + ChemComp(11);
    w_O  = ChemComp(2) + 2*ChemComp(4) + ChemComp(8) + 2*ChemComp(12) + ChemComp(13);
    w_N  = 2*ChemComp(3) + ChemComp(7) + ChemComp(9);
    w_Ar = ChemComp(5);
    w_H  = ChemComp(6) + ChemComp(10) + 2*ChemComp(14);
    ChemCompRe = [w_H w_He w_N w_O w_Ar]/sum([w_H w_He w_N w_O w_Ar]);
end