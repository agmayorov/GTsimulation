[x, y] = meshgrid(-2:0.02:2, -2:0.02:2);
x_ = reshape(x, 1, []);
y_ = reshape(y, 1, []);
figure;

HField = HelioBfield("NParam",struct());
[Bx_, By_, Bz_] = HField.GetBfield(x_, y_, 0, 0);

B_ = sqrt(Bx_.^2 + By_.^2 + Bz_.^2);
B = reshape(B_, size(x)); 
pcolor(x, y, atan(0.2.*B.*((x.^2+y.^2)>0.05.^2)));
shading interp;
cb=colorbar;
hold on;