function [beta] = BFGS(E, beta, B, eps)
% BFGS
%
% Input:
%   E    : objective function
%   beta : initial weight, a matrix with size = (h, m)
%   B    : initial approximation of inverse Hessian, size = (h*m, h*m)
%   eps  : accuracy for stopping criterion

global h  % number of hidden units
global m  % output dimension

iter = 0;
k = 0;
[v, g] = E(beta);
g = reshape(g, [h*m,1]);

while (norm(g) > eps)
    % Compute direction
    p = -B * g;

    % Compute step size
    gp = g' * p;
    p = reshape(p, [h,m]);
    a = BacktrackingLS(1, 0.9, 1e-4, beta, p, E, v, gp);

    beta_new = beta + a * p;

    [v_new, g_new] = E(beta_new);
    g_new = reshape(g_new, [h*m,1]);
    s = beta_new - beta; s = reshape(s, [h*m,1]);
    y = g_new - g;
    rho = y' * s;

    % For first iteration, need to update the initial B
    if (k == 0)
        B = (rho / (y' * y)) * eye(h*m);
    end

    % Compute the approximation of inverse Hessian B
    rho = 1 / rho;
    D = B * y * s';
    delta_B = rho * ((1 + rho * y' * B * y) * (s * s') - D - D');
    B = B + delta_B;

    iter = iter + 1;
    k = k + 1;
    beta = beta_new;
    v = v_new;
    g = g_new;
end

end


function a = BacktrackingLS(a, tau, c1, beta, p, E, phi0, gp)
% Perform backtracking line search to find step size
% that satisifes Armijo condition
%
% Input:
%   a    : initial step size, must be 1 for quasi-Newton
%   tau  : float in (0,1)
%   c1   : float in (0,1)
%   beta : current position, size = (h, m)
%   p    : direction, size = (h, m)
%   E    : objective function
%   phi0 : a scalar, which is E(\beta)
%   gp   : a scalar, which is \nabla E(\beta) * p

while true
    [phi, ~] = E(beta + a * p);
    if ( phi <= phi0 + c1 * a * gp )
        break;
    end
    a = a * tau; 
end

end
