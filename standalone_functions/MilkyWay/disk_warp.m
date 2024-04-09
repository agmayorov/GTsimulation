close all
clear

% w0 = @(R)  2.063e-4 * R.^3 - 0.002329 * R.^2 + 0.001374 * R - 0.03613;
% w1 = @(R) -6.838e-5 * R.^3 - 0.005807 * R.^2 + 0.063540 * R - 0.11340;
% w2 = @(R)  1.217e-4 * R.^3 + 0.001202 * R.^2 - 0.058400 * R + 0.26010;

% --- Fit parameters using .xml data ---

theta1 = 1.47548;
theta2 = 2.76487;

w0p = [ 2.063e-4, -0.002329,  0.001374, -0.03613];
w1p = [-6.838e-5, -0.005807,  0.063540, -0.11340];
w2p = [ 1.217e-4,  0.001202, -0.058400,  0.26010];

% --- Fit parameters using article data ---

% theta1 = 4.61;
% theta2 = 2.73;

% w0p = [ 2.075e-4, -0.002427,  0.003499, -0.04759];
% w1p = [ 6.192e-5,  0.006261, -0.070670,  0.14130];
% w2p = [ 1.057e-4,  0.001885, -0.066180,  0.28310];

w0 = @(R) polyval(w0p, R);
w1 = @(R) polyval(w1p, R);
w2 = @(R) polyval(w2p, R);

n = 101;
[x, y] = meshgrid(linspace(-20, 20, n), linspace(-20, 20, n));

[theta, rho] = cart2pol(x,y);
z = w0(rho) + w1(rho) .* sin(theta - theta1) + w2(rho) .* sin(2 * theta - theta2);

x(rho > 20) = NaN;
y(rho > 20) = NaN;
z(rho > 20) = NaN;

surf(x, y, z)
zlim([-5 5])
xlabel('X, kpc')
ylabel('Y, kpc')
zlabel('Z, kpc')

view(-30, 30)
