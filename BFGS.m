function [beta, errors] = BFGS(E, beta, B, eps, h, m, W, b, f, X, T, lambda, N)
% BFGS
%
% Input:
%   E    : objective function
%   beta : initial weight, a matrix with size = (h, m)
%   B    : initial approximation of inverse Hessian, size = (h*m, h*m)
%   eps  : accuracy for stopping criterion

iter = 0;
k = 0;
[v, g] = E(beta, X, T, W, b, N, f, lambda);
errors = [v];
g = g(:);

while (norm(g) > eps)
    % Compute direction
    p = -B * g;

    % Compute step size
    phid0 = g' * p;
    p = reshape(p, [h,m]);
    %[a, v_new, g_new] = BacktrackingLS(1, 0.9, 1e-4, beta, p, E, v, phid0);
    [a, v_new, g_new] = ArmijoWolfeLS(1, 0.9, 1e-4, 0.9, beta, p, E, v, phid0, X, T, W, b, N, f, lambda);

    beta_new = beta + a * p;

    g_new = g_new(:);
    s = beta_new - beta; s = s(:);
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
    errors = [errors, v];
end
fprintf('\n### BFGS ###\n')
fprintf('\n# iterations = %d\n\nFinal error = %d\n\n', iter, v);
% Plot
figure
scatter(1:iter+1, errors)
title('BFGS | Error function')
xlabel('iteration')
ylabel('Error')
all_decreasing = true;
% Test all decreasing errors
for i = 1:iter
    if errors(i) < errors(i+1)
        all_decreasing = false;
        break;
    end
end
if all_decreasing
    fprintf('The errors were all decreasing.\n\n')
else
    fprintf('The errors were *NOT* all decreasing.\n')
end
end


function [a, phi, g] = BacktrackingLS(a, tau, c1, beta, p, E, phi0, phid0)
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
%   phid0: a scalar, which is \nabla E(\beta) * p
%
% Output:
%   a    : optimal step size
%   phi  : value of objective function at new beta, E(beta + a * p)
%   g    : gradient at new beta, \nabla E(beta + a * p)

while true
    [phi, g] = E(beta + a * p);
    if ( phi <= phi0 + c1 * a * phid0 )
        break;
    end
    a = a * tau; 
end

end


function [a, phi, g] = ArmijoWolfeLS(a, tau, c1, c2, beta, p, E, phi0, phid0, X, T, W, b, N, f, lambda)
% Perform line search to find step size
% that satisifes Armijo-Wolfe condition
%
% Input:
%   a    : initial step size, must be 1 for quasi-Newton
%   tau  : float in (0,1)
%   c1   : float in (0,1)
%   c2   : float in (0,1), c1 < c2
%   beta : current position, size = (h, m)
%   p    : direction, size = (h, m)
%   E    : objective function
%   phi0 : a scalar, which is E(\beta)
%   phid0: a scalar, which is \nabla E(\beta) * p
%
% Output:
%   a    : optimal step size
%   phi  : value of objective function at new beta, E(beta + a * p)
%   g    : gradient at new beta, \nabla E(beta + a * p)

p_vec = p(:); % reshape matrix p as a vector

while true
    [phi, g] = E(beta + a * p, X, T, W, b, N, f, lambda);
    g = g(:);
    phid = g' * p_vec;
    if (phi <= phi0 + c1 * a * phid0) && (abs(phid) <= c2 * abs(phid0))
        return;
    end
    if (phid >= 0)
        break;
    end
    a = a / tau; 
end

a_min = 0;
a_max = a;
phid_min = phid0;
phid_max = phid;

while (a_max > a_min) && (phid > 1e-12)
    a = (a_min * phid_max - a_max * phid_min) / (phid_max - phid_min);
    a = max(a_min + 0.01 * (a_max - a_min), min(a_max - 0.01 * (a_max - a_min), a));
    [phi, g] = E(beta + a * p, X, T, W, b, N, f, lambda);
    g = g(:);
    phid = g' * p_vec;
    if (phi <= phi0 + c1 * a * phid0) && (abs(phid) <= c2 * abs(phid0))
        return;
    end
    if (phid < 0)
        a_min = a;
        phid_min = phid;
    else
        a_max = a;
        phid_max = phid;
    end
end

end
