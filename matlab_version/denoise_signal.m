%% denoise_signal.m
% This function implements a nonlinear filtering of eye-movement 
% time-series to reduce noise based on sparse time-series properties. 
% In the abasence of noise, the first-order and higher-order temporal 
% derivatives of eye-movement time-series can be modeled as sparse. Then, 
% from a noisy eye-movement time-series y, the time-series x can be estimated
% by minimizing the objective function: 
% f(x) = 1/2 ||y-x||^2 + alpha ||D1 x||1 + beta ||D3x||1.

% Reference: W. Dai, I. Selesnick, J.-R. Rizzo, J. Rucker and T. Hudson, 
% "Detection of normal and slow saccades using implicit piecewise polynomial 
% approximation," Journal of Vision, vol. 21, no. 6, pp. 1-18, 2021. 

function x = denoise_signal(position, alpha, beta, Nit)
% param position: time-series saccade position.
% param alpha: regularization parameter used for denoising.
% param beta: regularization parameter used for denoising
% param Nit: number of iterations.

% return x: position signal been denoised.

% Get the length of saccade position signal and create an array of ones
position = position(:);
N = length(position);
e = ones(N, 1);

% Smoothed penalty function
EPS = 1e-10;
psi = @(x) sqrt(x.^2 + EPS);

% Create the first-order difference and third-order difference operators
D1 = spdiags([-e, e], 0:1, N-1, N);
D3 = spdiags([-e 3*e -3*e e], 0:3, N-3, N);
I = spdiags(e, 0, N, N);

x = position;

% Implement the nonlinear filter to denoise the signal
for i = 1:Nit
    Lam1 = spdiags(alpha./psi(diff(x)), 0, N-1, N-1);
    Lam3 = spdiags(beta./psi(diff(x,3)), 0, N-3, N-3);
    temp = I + D1' * Lam1 * D1 + D3' * Lam3 * D3;
    x = temp \ position;
end

end