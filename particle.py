import sys

class Particle:

    def __init__(self, Type = None, Z = None, M = None, PDG = None, E = None):
        try:
            if E is None or E <= 0:
                raise SyntaxError
            else:
                self.T = E
                #First constructor connected with table
                if Z is None and M is None and PDG is None and Type is not None:
                    self.Type = Type
                    match Type:
                                case "gamma": self.A=0; self.Z=0; 	self.M=0; self.PDG=22;
                                case "gam": self.A=0; self.Z=0; 		self.M=0;				self.PDG=22;
                                case "pr":  self.A=1; self.Z=1; 		self.M=938.27208816;	 	self.PDG=2212;
                                case "proton": self.A=1; self.Z=1; 	self.M=938.27208816;	 	self.PDG=2212;
                                case "apr": self.A=1; self.Z=-1; 		self.M=938.27208816; 	self.PDG=-2212;
                                case "de":  self.A=2; self.Z=1; 		self.M=1875.61294257; 	self.PDG=1000010020;
                                case "deuteron": self.A=2; self.Z=1; 	self.M=1875.61294257; 	self.PDG=1000010020;
                                case "ade": self.A=2; self.Z=-1; 		self.M=1875.61294257; 	self.PDG=-1000010020;
                                case "tr":  self.A=3; self.Z=1; 		self.M=2808.92113298;	self.PDG=1000010030;
                                case "triton":  self.A=3; self.Z=1; 	self.M=2808.92113298;	self.PDG=1000010030;
                                case "atr": self.A=3; self.Z=-1; 		self.M=2808.92113298;	self.PDG=-1000010030;
                                case "he3": self.A=3; self.Z=2;		self.M=2808.391;			self.PDG=1000020030;
                                case "ahe3":self.A=3; self.Z=-2;		self.M=2808.391;			self.PDG=-1000020030;
                                case "he4": self.A=4; self.Z=2; 		self.M=3727.3794066;		self.PDG=1000020040;
                                case "alpha": self.A=4; self.Z=2; 	self.M=3727.3794066;		self.PDG=1000020040;
                                case "ahe4":self.A=4; self.Z=-2; 		self.M=3727.3794066;		self.PDG=-1000020040;
                                case "ele": self.A=0; self.Z=-1;		self.M=0.51099895000;	self.PDG=11;
                                case "pos": self.A=0; self.Z=1;		self.M=0.51099895000;	self.PDG=-11;
                                case "mup": self.A=0; self.Z=1; 		self.M=105.6583745;		self.PDG=13;
                                case "mum": self.A=0; self.Z=-1; 		self.M=105.6583745;		self.PDG=-13;
                                case "pip": self.A=0; self.Z=1; 		self.M=139.57039;		self.PDG=211;
                                case "pim": self.A=0; self.Z=-1; 		self.M=139.57039;		self.PDG=-211;
                                case "kap": self.A=0; self.Z=1; 		self.M=493.677;			self.PDG=321;
                                case "kam": self.A=0; self.Z=-1; 		self.M=493.677;			self.PDG=-321;
                                case "pio": self.A=0; self.Z=0; 		self.M=134.9768;			self.PDG=111;
                                case "n":   self.A=1; self.Z=0; 		self.M=939.565413;		self.PDG=2112;
                                case "neutron": self.A=1; self.Z=0; 	self.M=939.565413;		self.PDG=2112;
                                case "an":  self.A=1; self.Z=0; 		self.M=939.565413; 		self.PDG=-2112;

                                case "Li-6":  self.A=6; self.Z=3; 	self.M=5601.518;			self.PDG=1000030060; 		#Ab=7.59;		Thd=0;
                                case "Li-7":  self.A=7; self.Z=3; 	self.M=6533.833;			self.PDG=1000030070; 		#Ab=92.41;		Thd=0;

                                case "Be-7":  self.A=7; self.Z=4; 	self.M=6536.227;			self.PDG=1000040070; 		#Ab=0;			Thd=0.146;
                                case "Be-9":  self.A=9; self.Z=4; 	self.M=8392.749;			self.PDG=1000040090; 		#Ab=100;			Thd=0;
                                case "Be-10": self.A=10; self.Z=4; 	self.M=9327.547;			self.PDG=1000040100; 		#Ab=0;			Thd=1.51e6;

                                case "B-10":  self.A=10; self.Z=5; 	self.M=9324.435;			self.PDG=1000050100; 		#Ab=19.8;		Thd=0;
                                case "B-11":  self.A=11; self.Z=5; 	self.M=10252.547;		self.PDG=1000050110; 		#Ab=80.2;		Thd=0;

                                case "C-12":  self.A=12; self.Z=6; 	self.M=11174.862;		self.PDG=1000060120; 		#Ab=98.89;		Thd=0;
                                case "C-13":  self.A=13; self.Z=6; 	self.M=12109.481;		self.PDG=1000060130; 		#Ab=1.11;		Thd=0;

                                case "N-14":  self.A=14; self.Z=7; 	self.M=13040.202;		self.PDG=1000070140; 		#Ab=99.634;		Thd=0;
                                case "N-15":  self.A=15; self.Z=7; 	self.M=13968.934;		self.PDG=1000070150; 		#Ab=0.366;		Thd=0;

                                case "O-16":  self.A=16; self.Z=8; 	self.M=14895.079;		self.PDG=1000080160; 		#Ab=99.762;		Thd=0;
                                case "O-17":  self.A=17; self.Z=8; 	self.M=15830.501;		self.PDG=1000080170; 		#Ab=0.038;		Thd=0;
                                case "O-18":  self.A=18; self.Z=8; 	self.M=16762.023;		self.PDG=1000080180; 		#Ab=0.200;		Thd=0;

                                case "F-19":  self.A=19; self.Z=9; 	self.M=17692.300;		self.PDG=1000090190; 		#Ab=100;			Thd=0;

                                case "Ne-20": self.A=20; self.Z=10; 	self.M=18617.728;		self.PDG=1000100200; 		#Ab=90.48;		Thd=0;
                                case "Ne-21": self.A=21; self.Z=10; 	self.M=19550.532;		self.PDG=1000100210; 		#Ab=0.27;		Thd=0;
                                case "Ne-22": self.A=22; self.Z=10; 	self.M=20479.734;		self.PDG=1000100220; 		#Ab=9.25;		Thd=0;

                                case "Na-23": self.A=23; self.Z=11; 	self.M=21409.211;		self.PDG=1000110230; 		#Ab=100;			Thd=0;

                                case "Mg-24": self.A=24; self.Z=12; 	self.M=22335.791;		self.PDG=1000120240; 		#Ab=78.99;		Thd=0;
                                case "Mg-25": self.A=25; self.Z=12; 	self.M=23268.025;		self.PDG=1000120250; 		#Ab=10.00;		Thd=0;
                                case "Mg-26": self.A=26; self.Z=12; 	self.M=24196.497;		self.PDG=1000120260; 		#Ab=11.01;		Thd=0;

                                case "Al-27": self.A=27; self.Z=13; 	self.M=25126.499;		self.PDG=1000130270; 		#Ab=100;			Thd=0;

                                case "Si-28": self.A=28; self.Z=14; 	self.M=26053.185;		self.PDG=1000140280; 		#Ab=92.230;		Thd=0;
                                case "Si-29": self.A=29; self.Z=14; 	self.M=26984.277;		self.PDG=1000140290; 		#Ab=4.683;		Thd=0;
                                case "Si-30": self.A=30; self.Z=14; 	self.M=27913.233;		self.PDG=1000140300; 		#Ab=3.087;		Thd=0;

                                case "P-31":  self.A=31; self.Z=15; 	self.M=28844.209;		self.PDG=1000150310; 		#Ab=100;			Thd=0;

                                case "S-32":  self.A=32; self.Z=16; 	self.M=29773.617;		self.PDG=1000160320; 		#Ab=95.02;		Thd=0;
                                case "S-33":  self.A=33; self.Z=16; 	self.M=30704.540;		self.PDG=1000160330; 		#Ab=0.75;		Thd=0;
                                case "S-34":  self.A=34; self.Z=16; 	self.M=31632.689;		self.PDG=1000160340; 		#Ab=4.21;		Thd=0;
                                case "S-36":  self.A=36; self.Z=16; 	self.M=33494.944;		self.PDG=1000160360; 		#Ab=0.02;		Thd=0;

                                case "Cl-35": self.A=35; self.Z=17; 	self.M=32564.590;		self.PDG=1000170350; 		#Ab=75.77;		Thd=0;
                                case "Cl-37": self.A=37; self.Z=17; 	self.M=34424.829;		self.PDG=1000170370; 		#Ab=24.23;		Thd=0;

                                case "Ar-36": self.A=36; self.Z=18; 	self.M=33494.355;		self.PDG=1000180360; 		#Ab=0.3365;		Thd=0;
                                case "Ar-38": self.A=38; self.Z=18; 	self.M=35352.860;		self.PDG=1000180380; 		#Ab=0.0632;		Thd=0;
                                case "Ar-40": self.A=40; self.Z=18; 	self.M=37215.522;		self.PDG=1000180400; 		#Ab=99.6003;		Thd=0;

                                case "K-39":  self.A=39; self.Z=19; 	self.M=36284.750;		self.PDG=1000190390; 		#Ab=93.2581;		Thd=0;
                                case "K-40":  self.A=40; self.Z=19; 	self.M=37216.516;		self.PDG=1000190400; 		#Ab=0.0117;		Thd=1.248e9;
                                case "K-41":  self.A=41; self.Z=19; 	self.M=38145.986;		self.PDG=1000190410; 		#Ab=6.7302;		Thd=0;

                                case "Ca-40": self.A=40; self.Z=20; 	self.M=37214.694;		self.PDG=1000200400; 		#Ab=96.94;		Thd=3.0e21;
                                case "Ca-42": self.A=42; self.Z=20; 	self.M=39073.981;		self.PDG=1000200420; 		#Ab=0.647;		Thd=0;
                                case "Ca-43": self.A=43; self.Z=20; 	self.M=40005.614;		self.PDG=1000200430; 		#Ab=0.135;		Thd=0;
                                case "Ca-44": self.A=44; self.Z=20; 	self.M=40934.048;		self.PDG=1000200440; 		#Ab=2.09;		Thd=0;
                                case "Ca-46": self.A=46; self.Z=20; 	self.M=42795.369;		self.PDG=1000200460; 		#Ab=0.004;		Thd=0.28e16;
                                case "Ca-48": self.A=48; self.Z=20; 	self.M=44657.278;		self.PDG=1000200480; 		#Ab=0.187;		Thd=1.9e19;

                                case "Sc-45": self.A=45; self.Z=21; 	self.M=41865.432;		self.PDG=1000210450; 		#Ab=100;			Thd=0;

                                case "Ti-46": self.A=46; self.Z=22; 	self.M=42793.359;		self.PDG=1000220460; 		#Ab=8.25;		Thd=0;
                                case "Ti-47": self.A=47; self.Z=22; 	self.M=43724.044;		self.PDG=1000220470; 		#Ab=7.44;		Thd=0;
                                case "Ti-48": self.A=48; self.Z=22; 	self.M=44651.982;		self.PDG=1000220480; 		#Ab=73.72;		Thd=0;
                                case "Ti-49": self.A=49; self.Z=22; 	self.M=45583.406;		self.PDG=1000220490; 		#Ab=5.41;		Thd=0;
                                case "Ti-50": self.A=50; self.Z=22; 	self.M=46512.031;		self.PDG=1000220500; 		#Ab=5.18;		Thd=0;

                                case "V-50":  self.A=50; self.Z=23; 	self.M=46513.725;		self.PDG=1000230500; 		#Ab=0.250;		Thd=1.4e17;
                                case "V-51":  self.A=51; self.Z=23; 	self.M=47442.240;		self.PDG=1000230510; 		#Ab=99.750;		Thd=0;

                                case "Cr-50": self.A=50; self.Z=24; 	self.M=46512.177;		self.PDG=1000240500; 		#Ab=4.345;		Thd=1.3e18;
                                case "Cr-52": self.A=52; self.Z=24; 	self.M=48370.008;		self.PDG=1000240520; 		#Ab=83.789;		Thd=0;
                                case "Cr-53": self.A=53; self.Z=24; 	self.M=49301.633;		self.PDG=1000240530; 		#Ab=9.501;		Thd=0;
                                case "Cr-54": self.A=54; self.Z=24; 	self.M=50231.480;		self.PDG=1000240540; 		#Ab=2.36;		Thd=0;

                                case "Mn-55": self.A=55; self.Z=25; 	self.M=51161.685;		self.PDG=1000250550; 		#Ab=100;			Thd=0;

                                case "Fe-54": self.A=54; self.Z=26; 	self.M=50231.138;		self.PDG=1000260540; 		#Ab=5.845;		Thd=0;
                                case "Fe-56": self.A=56; self.Z=26; 	self.M=52089.773;		self.PDG=1000260560; 		#Ab=91.754;		Thd=0;
                                case "Fe-57": self.A=57; self.Z=26; 	self.M=53021.692;		self.PDG=1000260570; 		#Ab=2.119;		Thd=0;
                                case "Fe-58": self.A=58; self.Z=26; 	self.M=53951.213;		self.PDG=1000260580; 		#Ab=0.282;		Thd=0;

                                case "Co-59": self.A=59; self.Z=27; 	self.M=54882.121;		self.PDG=1000270590; 		#Ab=100;			Thd=0;

                                case "Ni-58": self.A=58; self.Z=28; 	self.M=53952.117;		self.PDG=1000280580; 		#Ab=68.077;		Thd=0;
                                case "Ni-60": self.A=60; self.Z=28; 	self.M=55810.860;		self.PDG=1000280600; 		#Ab=26.223;		Thd=0;
                                case "Ni-61": self.A=61; self.Z=28; 	self.M=56742.605;		self.PDG=1000280610; 		#Ab=1.140;		Thd=0;
                                case "Ni-62": self.A=62; self.Z=28; 	self.M=57671.574;		self.PDG=1000280620; 		#Ab=3.634;		Thd=0;
                                case "Ni-64": self.A=64; self.Z=28; 	self.M=59534.209;		self.PDG=1000280640; 		#Ab=0.926;		Thd=0;

                                case "Cu-63": self.A=63; self.Z=29; 	self.M=58603.724;		self.PDG=1000290630; 		#Ab=69.17;		Thd=0;
                                case "Cu-65": self.A=65; self.Z=29; 	self.M=60465.027;		self.PDG=1000290650; 		#Ab=30.83;		Thd=0;

                                case "Zn-64": self.A=64; self.Z=30; 	self.M=59534.283;		self.PDG=1000300640; 		#Ab=48.63;		Thd=0;
                                case "Zn-66": self.A=66; self.Z=30; 	self.M=61394.375;		self.PDG=1000300660; 		#Ab=27.90;		Thd=0;
                                case "Zn-67": self.A=67; self.Z=30; 	self.M=62326.888;		self.PDG=1000300670; 		#Ab=4.10;		Thd=0;
                                case "Zn-68": self.A=68; self.Z=30; 	self.M=63256.255;		self.PDG=1000300680; 		#Ab=18.75;		Thd=0;
                                case "Zn-70": self.A=70; self.Z=30; 	self.M=65119.685;		self.PDG=1000300700; 		#Ab=0.62;		Thd=1.3e16;

                                case "Ga-69": self.A=69; self.Z=31; 	self.M=64187.918;		self.PDG=1000310690; 		#Ab=60.108;		Thd=0;
                                case "Ga-71": self.A=71; self.Z=31; 	self.M=66050.093;		self.PDG=1000310710; 		#Ab=39.892;		Thd=0;

                                case "Ge-70": self.A=70; self.Z=32; 	self.M=65117.665;		self.PDG=1000320700; 		#Ab=20.37;		Thd=0;
                                case "Ge-72": self.A=72; self.Z=32; 	self.M=66978.630;		self.PDG=1000320720; 		#Ab=27.31;		Thd=0;
                                case "Ge-73": self.A=73; self.Z=32; 	self.M=67911.413;		self.PDG=1000320730; 		#Ab=7.76;		Thd=0;
                                case "Ge-74": self.A=74; self.Z=32; 	self.M=68840.782;		self.PDG=1000320740; 		#Ab=36.73;		Thd=0;
                                case "Ge-76": self.A=76; self.Z=32; 	self.M=70703.979;		self.PDG=1000320760; 		#Ab=7.83;		Thd=0;

                                case "As-75": self.A=75; self.Z=33; 	self.M=69772.155;		self.PDG=1000330750; 		#Ab=100;			Thd=0;

                                case "Se-74": self.A=74; self.Z=34; 	self.M=68840.970;		self.PDG=1000340740; 		#Ab=0.89;		Thd=0;
                                case "Se-76": self.A=76; self.Z=34; 	self.M=70700.918;		self.PDG=1000340760; 		#Ab=9.37;		Thd=0;
                                case "Se-77": self.A=77; self.Z=34; 	self.M=71633.065;		self.PDG=1000340770; 		#Ab=7.63;		Thd=0;
                                case "Se-78": self.A=78; self.Z=34; 	self.M=72562.132;		self.PDG=1000340780; 		#Ab=23.77;		Thd=0;
                                case "Se-80": self.A=80; self.Z=34; 	self.M=74424.386;		self.PDG=1000340800; 		#Ab=49.61;		Thd=0;
                                case "Se-82": self.A=82; self.Z=34; 	self.M=76287.540;		self.PDG=1000340820; 		#Ab=8.73;		Thd=0;

                                case "Br-79": self.A=79; self.Z=35; 	self.M=73494.073;		self.PDG=1000350790; 		#Ab=50.69;		Thd=0;
                                case "Br-81": self.A=81; self.Z=35; 	self.M=75355.154;		self.PDG=1000350810; 		#Ab=49.31;		Thd=0;

                                case "Kr-78": self.A=78; self.Z=36; 	self.M=72563.956;		self.PDG=1000360780; 		#Ab=0.35;		Thd=2.3e20;
                                case "Kr-80": self.A=80; self.Z=36; 	self.M=74423.232;		self.PDG=1000360800; 		#Ab=2.28;		Thd=0;
                                case "Kr-82": self.A=82; self.Z=36; 	self.M=76283.523;		self.PDG=1000360820; 		#Ab=11.58;		Thd=0;
                                case "Kr-83": self.A=83; self.Z=36; 	self.M=77215.624;		self.PDG=1000360830; 		#Ab=11.49;		Thd=0;
                                case "Kr-84": self.A=84; self.Z=36; 	self.M=78144.669;		self.PDG=1000360840; 		#Ab=57.00;		Thd=0;
                                case "Kr-86": self.A=86; self.Z=36; 	self.M=80006.822;		self.PDG=1000360860; 		#Ab=17.30;		Thd=0;

                                case "Rb-85": self.A=85; self.Z=37; 	self.M=79075.916;		self.PDG=1000370850; 		#Ab=72.17;		Thd=0;
                                case "Rb-87": self.A=87; self.Z=37; 	self.M=80936.474;		self.PDG=1000370870; 		#Ab=27.83;		Thd=4.81e10;

                                case "Sr-84": self.A=84; self.Z=38; 	self.M=78145.434;		self.PDG=1000380840; 		#Ab=0.56;		Thd=0;
                                case "Sr-86": self.A=86; self.Z=38; 	self.M=80004.542;		self.PDG=1000380860; 		#Ab=9.86;		Thd=0;
                                case "Sr-87": self.A=87; self.Z=38; 	self.M=80935.680;		self.PDG=1000380870; 		#Ab=7.00;		Thd=0;
                                case "Sr-88": self.A=88; self.Z=38; 	self.M=81864.132;		self.PDG=1000380880; 		#Ab=82.58;		Thd=0;

                                case "Y-89":  self.A=89; self.Z=39; 	self.M=82795.336;		self.PDG=1000390890; 		#Ab=100;			Thd=0;

                                case "Zr-90": self.A=90; self.Z=40; 	self.M=83725.253;		self.PDG=1000400900; 		#Ab=51.45;		Thd=0;
                                case "Zr-91": self.A=91; self.Z=40; 	self.M=84657.624;		self.PDG=1000400910; 		#Ab=11.22;		Thd=0;
                                case "Zr-92": self.A=92; self.Z=40; 	self.M=85588.554;		self.PDG=1000400920; 		#Ab=17.15;		Thd=0;
                                case "Zr-94": self.A=94; self.Z=40; 	self.M=87452.730;		self.PDG=1000400940; 		#Ab=17.38;		Thd=0;
                                case "Zr-96": self.A=96; self.Z=40; 	self.M=89317.542;		self.PDG=1000400960; 		#Ab=2.80;		Thd=2.0e19;

                                case "Nb-93": self.A=93; self.Z=41; 	self.M=86520.783;		self.PDG=1000410930; 		#Ab=100;			Thd=0;

                                case "Mo-92": self.A=92; self.Z=42; 	self.M=85589.181;		self.PDG=1000420920; 		#Ab=14.84;		Thd=0;
                                case "Mo-94": self.A=94; self.Z=42; 	self.M=87450.565;		self.PDG=1000420940; 		#Ab=9.25;		Thd=0;
                                case "Mo-95": self.A=95; self.Z=42; 	self.M=88382.761;		self.PDG=1000420950; 		#Ab=15.92;		Thd=0;
                                case "Mo-96": self.A=96; self.Z=42; 	self.M=89313.172;		self.PDG=1000420960; 		#Ab=16.68;		Thd=0;
                                case "Mo-97": self.A=97; self.Z=42; 	self.M=90245.916;		self.PDG=1000420970; 		#Ab=9.55;		Thd=0;
                                case "Mo-98": self.A=98; self.Z=42; 	self.M=91176.839;		self.PDG=1000420980; 		#Ab=24.13;		Thd=0;
                                case "Mo-100":self.A=100; self.Z=42; 	self.M=93041.754;		self.PDG=1000421000; 		#Ab=9.63;		Thd=7.3e18;

                                case "Ru-96":  self.A=96;  self.Z=44; self.M=89314.868;		self.PDG=1000440960; 		#Ab=5.54;		Thd=0;
                                case "Ru-98":  self.A=98;  self.Z=44; self.M=91175.704;		self.PDG=1000440980; 		#Ab=1.87;		Thd=0;
                                case "Ru-99":  self.A=99;  self.Z=44; self.M=92107.806;		self.PDG=1000440990; 		#Ab=12.76;		Thd=0;
                                case "Ru-100": self.A=100; self.Z=44; self.M=93037.698;		self.PDG=1000441000; 		#Ab=12.60;		Thd=0;
                                case "Ru-101": self.A=101; self.Z=44; self.M=93970.461;		self.PDG=1000441010; 		#Ab=17.06;		Thd=0;
                                case "Ru-102": self.A=102; self.Z=44; self.M=94900.806;		self.PDG=1000441020; 		#Ab=31.55;		Thd=0;
                                case "Ru-104": self.A=104; self.Z=44; self.M=96764.804;		self.PDG=1000441040; 		#Ab=18.62;		Thd=0;

                                case "Rh-103": self.A=103; self.Z=45; self.M=95832.865;		self.PDG=1000451030; 		#Ab=100;			Thd=0;

                                case "Pd-102": self.A=102; self.Z=46; self.M=94900.957;		self.PDG=1000461020; 		#Ab=1.02;		Thd=0;
                                case "Pd-104": self.A=104; self.Z=46; self.M=96762.480;		self.PDG=1000461040; 		#Ab=11.14;		Thd=0;
                                case "Pd-105": self.A=105; self.Z=46; self.M=97694.952;		self.PDG=1000461050; 		#Ab=22.33;		Thd=0;
                                case "Pd-106": self.A=106; self.Z=46; self.M=98624.956;		self.PDG=1000461060; 		#Ab=27.33;		Thd=0;
                                case "Pd-108": self.A=108; self.Z=46; self.M=100488.322;		self.PDG=1000461080; 		#Ab=26.46;		Thd=0;
                                case "Pd-110": self.A=110; self.Z=46; self.M=102352.485;		self.PDG=1000461100; 		#Ab=11.72;		Thd=0;

                                case "Ag-107": self.A=107; self.Z=47; self.M=99557.440;		self.PDG=1000471070; 		#Ab=51.839;		Thd=0;
                                case "Ag-109": self.A=109; self.Z=47; self.M=101420.107;		self.PDG=1000471090; 		#Ab=48.161;		Thd=0;

                                case "Cd-106": self.A=106; self.Z=48; self.M=98626.704;		self.PDG=1000481060; 		#Ab=1.25;		Thd=0;
                                case "Cd-108": self.A=108; self.Z=48; self.M=100487.572;		self.PDG=1000481080; 		#Ab=0.89;		Thd=0;
                                case "Cd-110": self.A=110; self.Z=48; self.M=102349.460;		self.PDG=1000481100; 		#Ab=12.49;		Thd=0;
                                case "Cd-111": self.A=111; self.Z=48; self.M=103282.049;		self.PDG=1000481110; 		#Ab=12.80;		Thd=0;
                                case "Cd-112": self.A=112; self.Z=48; self.M=104212.220;		self.PDG=1000481120; 		#Ab=24.13;		Thd=0;
                                case "Cd-113": self.A=113; self.Z=48; self.M=105145.245;		self.PDG=1000481130; 		#Ab=12.22;		Thd=7.7e15;
                                case "Cd-114": self.A=114; self.Z=48; self.M=106075.768;		self.PDG=1000481140; 		#Ab=28.73;		Thd=0;
                                case "Cd-116": self.A=116; self.Z=48; self.M=107940.057;		self.PDG=1000481160; 		#Ab=7.49;		Thd=3.1e19;

                                case "In-113": self.A=113; self.Z=49; self.M=105144.413;		self.PDG=1000491130; 		#Ab=4.29;		Thd=0;
                                case "In-115": self.A=115; self.Z=49; self.M=107007.234;		self.PDG=1000491150; 		#Ab=95.71;		Thd=4.41e14;

                                case "Sn-112": self.A=112; self.Z=50; self.M=104213.117;		self.PDG=1000501120; 		#Ab=0.97;		Thd=0;
                                case "Sn-114": self.A=114; self.Z=50; self.M=106074.206;		self.PDG=1000501140; 		#Ab=0.66;		Thd=0;
                                case "Sn-115": self.A=115; self.Z=50; self.M=107006.225;		self.PDG=1000501150; 		#Ab=0.34;		Thd=0;
                                case "Sn-116": self.A=116; self.Z=50; self.M=107936.226;		self.PDG=1000501160; 		#Ab=14.54;		Thd=0;
                                case "Sn-117": self.A=117; self.Z=50; self.M=108868.849;		self.PDG=1000501170; 		#Ab=7.68;		Thd=0;
                                case "Sn-118": self.A=118; self.Z=50; self.M=109799.086;		self.PDG=1000501180; 		#Ab=24.22;		Thd=0;
                                case "Sn-119": self.A=119; self.Z=50; self.M=110732.168;		self.PDG=1000501190; 		#Ab=8.59;		Thd=0;
                                case "Sn-120": self.A=120; self.Z=50; self.M=111662.625;		self.PDG=1000501200; 		#Ab=32.58;		Thd=0;
                                case "Sn-122": self.A=122; self.Z=50; self.M=113526.773;		self.PDG=1000501220; 		#Ab=4.63;		Thd=0;
                                case "Sn-124": self.A=124; self.Z=50; self.M=115391.470;		self.PDG=1000501240; 		#Ab=5.79;		Thd=0;

                                case "Sb-121": self.A=121; self.Z=51; self.M=112595.118;		self.PDG=1000511210; 		#Ab=57.21;		Thd=0;
                                case "Sb-123": self.A=123; self.Z=51; self.M=114458.477;		self.PDG=1000511230; 		#Ab=42.79;		Thd=0;

                                case "Te-120": self.A=120; self.Z=52; self.M=111663.304;		self.PDG=1000521200; 		#Ab=0.09;		Thd=0;
                                case "Te-122": self.A=122; self.Z=52; self.M=113525.382;		self.PDG=1000521220; 		#Ab=2.55;		Thd=0;
                                case "Te-123": self.A=123; self.Z=52; self.M=114458.018;		self.PDG=1000521230; 		#Ab=0.89;		Thd=9.2e16;
                                case "Te-124": self.A=124; self.Z=52; self.M=115388.160;		self.PDG=1000521240; 		#Ab=4.74;		Thd=0;
                                case "Te-125": self.A=125; self.Z=52; self.M=116321.156;		self.PDG=1000521250; 		#Ab=7.07;		Thd=0;
                                case "Te-126": self.A=126; self.Z=52; self.M=117251.608;		self.PDG=1000521260; 		#Ab=18.84;		Thd=0;
                                case "Te-128": self.A=128; self.Z=52; self.M=119115.668;		self.PDG=1000521280; 		#Ab=31.74;		Thd=8.8e18;
                                case "Te-130": self.A=130; self.Z=52; self.M=120980.297;		self.PDG=1000521300; 		#Ab=34.08;		Thd=5e23;

                                case "I-127":  self.A=127; self.Z=53; self.M=118183.672;		self.PDG=1000531270; 		#Ab=100;			Thd=0;

                                case "Xe-124": self.A=124; self.Z=54; self.M=115390.002;		self.PDG=1000541240; 		#Ab=0.095;		Thd=1.6e14;
                                case "Xe-126": self.A=126; self.Z=54; self.M=117251.482;		self.PDG=1000541260; 		#Ab=0.089;		Thd=0;
                                case "Xe-128": self.A=128; self.Z=54; self.M=119113.778;		self.PDG=1000541280; 		#Ab=1.910;		Thd=0;
                                case "Xe-129": self.A=129; self.Z=54; self.M=120046.435;		self.PDG=1000541290; 		#Ab=26.40;		Thd=0;
                                case "Xe-130": self.A=130; self.Z=54; self.M=120976.744;		self.PDG=1000541300; 		#Ab=4.071;		Thd=0;
                                case "Xe-131": self.A=131; self.Z=54; self.M=121909.705;		self.PDG=1000541310; 		#Ab=21.232;		Thd=0;
                                case "Xe-132": self.A=132; self.Z=54; self.M=122840.334;		self.PDG=1000541320; 		#Ab=26.909;		Thd=0;
                                case "Xe-134": self.A=134; self.Z=54; self.M=124704.478;		self.PDG=1000541340; 		#Ab=10.436;		Thd=5.8e22;
                                case "Xe-136": self.A=136; self.Z=54; self.M=126569.165;		self.PDG=1000541360; 		#Ab=8.857;		Thd=2.4e21;

                                case "Cs-133": self.A=133; self.Z=55; self.M=123772.526;		self.PDG=1000551330; 		#Ab=100;			Thd=0;

                                case "Ba-130": self.A=130; self.Z=56; self.M=120978.343;		self.PDG=1000561300; 		#Ab=0.106;		Thd=0;
                                case "Ba-132": self.A=132; self.Z=56; self.M=122840.157;		self.PDG=1000561320; 		#Ab=0.101;		Thd=3.0e21;
                                case "Ba-134": self.A=134; self.Z=56; self.M=124702.630;		self.PDG=1000561340; 		#Ab=2.417;		Thd=0;
                                case "Ba-135": self.A=135; self.Z=56; self.M=125635.224;		self.PDG=1000561350; 		#Ab=6.592;		Thd=0;
                                case "Ba-136": self.A=136; self.Z=56; self.M=126565.681;		self.PDG=1000561360; 		#Ab=7.854;		Thd=0;
                                case "Ba-137": self.A=137; self.Z=56; self.M=127498.341;		self.PDG=1000561370; 		#Ab=11.232;		Thd=0;
                                case "Ba-138": self.A=138; self.Z=56; self.M=128429.294;		self.PDG=1000561380; 		#Ab=71.698;		Thd=0;

                                case "La-138": self.A=138; self.Z=57; self.M=128430.520;		self.PDG=1000571380; 		#Ab=0.090;		Thd=1.02e11;
                                case "La-139": self.A=139; self.Z=57; self.M=129361.308;		self.PDG=1000571390; 		#Ab=99.910;		Thd=0;

                                case "Ce-136": self.A=136; self.Z=58; self.M=126567.078;		self.PDG=1000581360; 		#Ab=0.185;		Thd=0.7e14;
                                case "Ce-138": self.A=138; self.Z=58; self.M=128428.966;		self.PDG=1000581380; 		#Ab=0.251;		Thd=0.9e14;
                                case "Ce-140": self.A=140; self.Z=58; self.M=130291.439;		self.PDG=1000581400; 		#Ab=88.450;		Thd=0;
                                case "Ce-142": self.A=142; self.Z=58; self.M=132157.972;		self.PDG=1000581420; 		#Ab=11.114;		Thd=5e16;

                                case "Pr-141": self.A=141; self.Z=59; self.M=131224.484;		self.PDG=1000591410; 		#Ab=100;			Thd=0;

                                case "Nd-142": self.A=142; self.Z=60; self.M=132155.533;		self.PDG=1000601420; 		#Ab=27.2;		Thd=0;
                                case "Nd-143": self.A=143; self.Z=60; self.M=133088.975;		self.PDG=1000601430; 		#Ab=12.2;		Thd=0;
                                case "Nd-144": self.A=144; self.Z=60; self.M=134020.723;		self.PDG=1000601440; 		#Ab=23.8;		Thd=2.29e15;
                                case "Nd-145": self.A=145; self.Z=60; self.M=134954.533;		self.PDG=1000601450; 		#Ab=8.3;			Thd=0;
                                case "Nd-146": self.A=146; self.Z=60; self.M=135886.533;		self.PDG=1000601460; 		#Ab=17.2;		Thd=0;
                                case "Nd-148": self.A=148; self.Z=60; self.M=137753.039;		self.PDG=1000601480; 		#Ab=5.7;			Thd=0;
                                case "Nd-150": self.A=150; self.Z=60; self.M=139619.750;		self.PDG=1000601500; 		#Ab=5.6;			Thd=0.79e19;

                                case "Sm-144": self.A=144; self.Z=62; self.M=134021.482;		self.PDG=1000621440; 		#Ab=3.07;		Thd=0;
                                case "Sm-147": self.A=147; self.Z=62; self.M=136818.664;		self.PDG=1000621470; 		#Ab=14.99;		Thd=1.06e11;
                                case "Sm-148": self.A=148; self.Z=62; self.M=137750.088;		self.PDG=1000621480; 		#Ab=11.24;		Thd=7e15;
                                case "Sm-149": self.A=149; self.Z=62; self.M=138683.782;		self.PDG=1000621490; 		#Ab=13.82;		Thd=0;
                                case "Sm-150": self.A=150; self.Z=62; self.M=139615.361;		self.PDG=1000621500; 		#Ab=7.38;		Thd=0;
                                case "Sm-152": self.A=152; self.Z=62; self.M=141480.638;		self.PDG=1000621520; 		#Ab=26.75;		Thd=0;
                                case "Sm-154": self.A=154; self.Z=62; self.M=143345.932;		self.PDG=1000621540; 		#Ab=22.75;		Thd=0;

                                case "Eu-151": self.A=151; self.Z=63; self.M=140548.742;		self.PDG=1000631510; 		#Ab=47.81;		Thd=1.7e18;
                                case "Eu-153": self.A=153; self.Z=63; self.M=142413.016;		self.PDG=1000631530; 		#Ab=52.19;		Thd=0;

                                case "Gd-152": self.A=152; self.Z=64; self.M=141479.670;		self.PDG=1000641520; 		#Ab=0.20;		Thd=1.08e14;
                                case "Gd-154": self.A=154; self.Z=64; self.M=143343.659;		self.PDG=1000641540; 		#Ab=2.18;		Thd=0;
                                case "Gd-155": self.A=155; self.Z=64; self.M=144276.789;		self.PDG=1000641550; 		#Ab=14.80;		Thd=0;
                                case "Gd-156": self.A=156; self.Z=64; self.M=145207.818;		self.PDG=1000641560; 		#Ab=20.47;		Thd=0;
                                case "Gd-157": self.A=157; self.Z=64; self.M=146141.023;		self.PDG=1000641570; 		#Ab=15.65;		Thd=0;
                                case "Gd-158": self.A=158; self.Z=64; self.M=147072.652;		self.PDG=1000641580; 		#Ab=24.84;		Thd=0;
                                case "Gd-160": self.A=160; self.Z=64; self.M=148938.387;		self.PDG=1000641600; 		#Ab=21.86;		Thd=3.1e19;

                                case "Tb-159": self.A=159; self.Z=65; self.M=148004.792;		self.PDG=1000651590; 		#Ab=100;			Thd=0;

                                case "Dy-156": self.A=156; self.Z=66; self.M=145208.808;		self.PDG=1000661560; 		#Ab=0.06;		Thd=0;
                                case "Dy-158": self.A=158; self.Z=66; self.M=147071.914;		self.PDG=1000661580; 		#Ab=0.10;		Thd=0;
                                case "Dy-160": self.A=160; self.Z=66; self.M=148935.636;		self.PDG=1000661600; 		#Ab=2.34;		Thd=0;
                                case "Dy-161": self.A=161; self.Z=66; self.M=149868.747;		self.PDG=1000661610; 		#Ab=18.91;		Thd=0;
                                case "Dy-162": self.A=162; self.Z=66; self.M=150800.116;		self.PDG=1000661620; 		#Ab=25.51;		Thd=0;
                                case "Dy-163": self.A=163; self.Z=66; self.M=151733.410;		self.PDG=1000661630; 		#Ab=24.90;		Thd=0;
                                case "Dy-164": self.A=164; self.Z=66; self.M=152665.317;		self.PDG=1000661640; 		#Ab=28.18;		Thd=0;

                                case "Ho-165": self.A=165; self.Z=67; self.M=153597.368;		self.PDG=1000671650; 		#Ab=100;			Thd=0;

                                case "Er-162": self.A=162; self.Z=68; self.M=150800.938;		self.PDG=1000681620; 		#Ab=0.139;		Thd=0;
                                case "Er-164": self.A=164; self.Z=68; self.M=152664.318;		self.PDG=1000681640; 		#Ab=1.601;		Thd=0;
                                case "Er-166": self.A=166; self.Z=68; self.M=154528.325;		self.PDG=1000681660; 		#Ab=33.503;		Thd=0;
                                case "Er-167": self.A=167; self.Z=68; self.M=155461.453;		self.PDG=1000681670; 		#Ab=22.869;		Thd=0;
                                case "Er-168": self.A=168; self.Z=68; self.M=156393.247;		self.PDG=1000681680; 		#Ab=26.978;		Thd=0;
                                case "Er-170": self.A=170; self.Z=68; self.M=158259.117;		self.PDG=1000681700; 		#Ab=14.910;		Thd=0;

                                case "Tm-169": self.A=169; self.Z=69; self.M=157325.948;		self.PDG=1000691690; 		#Ab=100;			Thd=0;

                                case "Yb-168": self.A=168; self.Z=70; self.M=156393.647;		self.PDG=1000701680; 		#Ab=0.13;		Thd=0;
                                case "Yb-170": self.A=170; self.Z=70; self.M=158257.442;		self.PDG=1000701700; 		#Ab=3.04;		Thd=0;
                                case "Yb-171": self.A=171; self.Z=70; self.M=159190.392;		self.PDG=1000701710; 		#Ab=14.28;		Thd=0;
                                case "Yb-172": self.A=172; self.Z=70; self.M=160121.938;		self.PDG=1000701720; 		#Ab=21.83;		Thd=0;
                                case "Yb-173": self.A=173; self.Z=70; self.M=161055.136;		self.PDG=1000701730; 		#Ab=16.13;		Thd=0;
                                case "Yb-174": self.A=174; self.Z=70; self.M=161987.237;		self.PDG=1000701740; 		#Ab=31.83;		Thd=0;
                                case "Yb-176": self.A=176; self.Z=70; self.M=163853.680;		self.PDG=1000701760; 		#Ab=12.76;		Thd=0;

                                case "Lu-175": self.A=175; self.Z=71; self.M=162919.998;		self.PDG=1000711750; 		#Ab=97.41;		Thd=0;
                                case "Lu-176": self.A=176; self.Z=71; self.M=163853.276;		self.PDG=1000711760; 		#Ab=2.59;		Thd=3.76e10;

                                case "Hf-174": self.A=174; self.Z=72; self.M=161987.317;		self.PDG=1000721740; 		#Ab=0.16;		Thd=2.0e15;
                                case "Hf-176": self.A=176; self.Z=72; self.M=163851.575;		self.PDG=1000721760; 		#Ab=5.26;		Thd=0;
                                case "Hf-177": self.A=177; self.Z=72; self.M=164784.757;		self.PDG=1000721770; 		#Ab=18.60;		Thd=0;
                                case "Hf-178": self.A=178; self.Z=72; self.M=165716.696;		self.PDG=1000721780; 		#Ab=27.28;		Thd=0;
                                case "Hf-179": self.A=179; self.Z=72; self.M=166650.163;		self.PDG=1000721790; 		#Ab=13.62;		Thd=0;
                                case "Hf-180": self.A=180; self.Z=72; self.M=167582.340;		self.PDG=1000721800; 		#Ab=35.08;		Thd=0;

                                case "Ta-181": self.A=181; self.Z=73; self.M=168514.670;		self.PDG=1000731810; 		#Ab=99.988;		Thd=0;

                                case "W-180":  self.A=180; self.Z=74; self.M=167581.462;		self.PDG=1000741800; 		#Ab=0.12;		Thd=1.8e18;
                                case "W-182":  self.A=182; self.Z=74; self.M=169445.847;		self.PDG=1000741820; 		#Ab=26.50;		Thd=8.3e18;
                                case "W-183":  self.A=183; self.Z=74; self.M=170379.221;		self.PDG=1000741830; 		#Ab=14.31;		Thd=1.3e19;
                                case "W-184":  self.A=184; self.Z=74; self.M=171311.375;		self.PDG=1000741840; 		#Ab=30.64;		Thd=2.9e19;
                                case "W-186":  self.A=186; self.Z=74; self.M=173177.561;		self.PDG=1000741860; 		#Ab=28.43;		Thd=2.7e19;

                                case "Re-185": self.A=185; self.Z=75; self.M=172244.243;		self.PDG=1000751850; 		#Ab=37.40;		Thd=0;
                                case "Re-187": self.A=187; self.Z=75; self.M=174109.837;		self.PDG=1000751870; 		#Ab=62.60;		Thd=4.12e10;

                                case "Os-184": self.A=184; self.Z=76; self.M=171311.804;		self.PDG=1000761840; 		#Ab=0.02;		Thd=5.6e13;
                                case "Os-186": self.A=186; self.Z=76; self.M=173176.049;		self.PDG=1000761860; 		#Ab=1.59;		Thd=2.0e15;
                                case "Os-187": self.A=187; self.Z=76; self.M=174109.324;		self.PDG=1000761870; 		#Ab=1.6;			Thd=0;
                                case "Os-188": self.A=188; self.Z=76; self.M=175040.900;		self.PDG=1000761880; 		#Ab=13.29;		Thd=0;
                                case "Os-189": self.A=189; self.Z=76; self.M=175974.545;		self.PDG=1000761890; 		#Ab=16.21;		Thd=0;
                                case "Os-190": self.A=190; self.Z=76; self.M=176906.318;		self.PDG=1000761900; 		#Ab=26.36;		Thd=0;
                                case "Os-192": self.A=192; self.Z=76; self.M=178772.132;		self.PDG=1000761920; 		#Ab=40.93;		Thd=0;

                                case "Ir-191": self.A=191; self.Z=77; self.M=177839.301;		self.PDG=1000771910; 		#Ab=37.3;		Thd=0;
                                case "Ir-193": self.A=193; self.Z=77; self.M=179704.462;		self.PDG=1000771930; 		#Ab=62.7;		Thd=0;

                                case "Pt-190": self.A=190; self.Z=78; self.M=176906.679;		self.PDG=1000781900; 		#Ab=0.014;		Thd=6.5e11;
                                case "Pt-192": self.A=192; self.Z=78; self.M=178770.698;		self.PDG=1000781920; 		#Ab=0.782;		Thd=0;
                                case "Pt-194": self.A=194; self.Z=78; self.M=180635.215;		self.PDG=1000781940; 		#Ab=32.967;		Thd=0;
                                case "Pt-195": self.A=195; self.Z=78; self.M=181568.676;		self.PDG=1000781950; 		#Ab=33.832;		Thd=0;
                                case "Pt-196": self.A=196; self.Z=78; self.M=182500.319;		self.PDG=1000781960; 		#Ab=25.242;		Thd=0;
                                case "Pt-198": self.A=198; self.Z=78; self.M=184366.047;		self.PDG=1000781980; 		#Ab=7.163;		Thd=0;

                                case "Au-197": self.A=197; self.Z=79; self.M=183432.808;		self.PDG=1000791970; 		#Ab=100;			Thd=0;

                                case "Hg-196": self.A=196; self.Z=80; self.M=182500.118;		self.PDG=1000801960; 		#Ab=0.15;		Thd=0;
                                case "Hg-198": self.A=198; self.Z=80; self.M=184363.978;		self.PDG=1000801980; 		#Ab=9.97;		Thd=0;
                                case "Hg-199": self.A=199; self.Z=80; self.M=185296.879;		self.PDG=1000801990; 		#Ab=16.87;		Thd=0;
                                case "Hg-200": self.A=200; self.Z=80; self.M=186228.416;		self.PDG=1000802000; 		#Ab=23.10;		Thd=0;
                                case "Hg-201": self.A=201; self.Z=80; self.M=187161.751;		self.PDG=1000802010; 		#Ab=13.18;		Thd=0;
                                case "Hg-202": self.A=202; self.Z=80; self.M=188093.563;		self.PDG=1000802020; 		#Ab=29.86;		Thd=0;
                                case "Hg-204": self.A=204; self.Z=80; self.M=189959.206;		self.PDG=1000802040; 		#Ab=6.87;		Thd=0;

                                case "Tl-203": self.A=203; self.Z=81; self.M=189026.130;		self.PDG=1000812030; 		#Ab=29.524;		Thd=0;
                                case "Tl-205": self.A=205; self.Z=81; self.M=190891.058;		self.PDG=1000812050; 		#Ab=70.476;		Thd=0;

                                case "Pb-204": self.A=204; self.Z=82; self.M=189957.765;		self.PDG=1000822040; 		#Ab=1.4;			Thd=1.4e17;
                                case "Pb-206": self.A=206; self.Z=82; self.M=191822.077;		self.PDG=1000822060; 		#Ab=24.1;		Thd=0;
                                case "Pb-207": self.A=207; self.Z=82; self.M=192754.905;		self.PDG=1000822070; 		#Ab=22.1;		Thd=0;
                                case "Pb-208": self.A=208; self.Z=82; self.M=193687.102;		self.PDG=1000822080; 		#Ab=52.4;		Thd=0;

                                case "Bi-209": self.A=209; self.Z=83; self.M=194621.575;		self.PDG=1000832090; 		#Ab=100;			Thd=0;

                                case "Th-232": self.A=232; self.Z=90; self.M=216096.066;		self.PDG=1000902320; 		#Ab=100;			Thd=1.4e10;

                                case "U-234":  self.A=234; self.Z=92; self.M=217960.730;		self.PDG=1000922340; 		#Ab=0.0054;		Thd=2.455e5;
                                case "U-235":  self.A=235; self.Z=92; self.M=218894.999;		self.PDG=1000922350; 		#Ab=0.7204;		Thd=7.04e8;
                                case "U-238":  self.A=238; self.Z=92; self.M=221695.869;		self.PDG=1000922380; 		#Ab=99.2742;		Thd=4.468e9;
                                case _:
                                        raise SyntaxError

                #Second constructor connected with personal parameters
                elif Z is not None and M is not None and Type is None:
                    self.Z = Z
                    self.M = M
                    self.PDG = PDG
                else:
                    raise SyntaxError
        except SyntaxError:
            print('Invalid syntax in constructor')
            sys.exit(1)

    def getZ(self):
        return self.Z

    def getM(self):
        return self.M

    def getPDG(self):
        return self.PDG

    def getT(self):
        return self.T
