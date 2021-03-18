function [beta, errors, tEnd, prec_tEnd] = BFGS(E, beta, B, eps, h, m, W, b, f, X, T, lambda, N, line_search, print_stat, opt_val, precision, keep_going_until_precision)
% BFGS
%
% Input:
%   E             : objective function
%   beta          : initial weight, a matrix with size = (h, m)
%   B             : initial approximation of inverse Hessian, size = (h*m, h*m)
%   eps           : accuracy for stopping criterion
%   h, m, W, b, f : Defines the ELM
%   X             : Training data inputs
%   T             : Training data outputs
%   lambda        : L2 regularization parameter
%   N             : Number of training samples
%   line_search   : Inexact line search method, can be either 'AWLS' or 'BLS'
%   print_stat    : boolean flag, when equal to true it print info
%   opt_val       : Optimal value of the problem, only used when
%                   'keep_going_until_precision' equals to true
%   precision     : Sets prefered relative error, keep the iterations going
%                   until the current relative error is greater than this 'precision'; only used when
%                   'keep_going_until_precision' equals to true
%   keep_going_until_precision : When it is equal to false it uses the
%                                relative norm of the gradient as a stopping condition, otherwise it
%                                uses the 'precision' condition.
%
% Outputs:
%   beta      : Optimal beta found by the optimization algorithm
%   errors    : List of the absolute errors computed at each iteration
%   tEnd      : Total number of seconds taken to found the optimal 'beta'.
%   prec_tEnd : Total number of seconds taken to reach 'precision'.

iter = 0;
k = 0;
[v, g] = E(beta, X, T, W, b, N, f, lambda);
errors = (v);
g = g(:);
ng0 = norm(g);

tStart = tic;
prec_tStart = tic;
prec_tEnd = 0;
got_precision = false;
if keep_going_until_precision == false
    prec_tEnd = toc(prec_tStart);
end

MAX_ITER = 500000;
while iter <= MAX_ITER && ((keep_going_until_precision == false && norm(g) > eps * ng0) || (keep_going_until_precision == true && got_precision == false))
    % If B is NAN, then beta does not change, so we can stop
    if isnan(B)
        break
    end

    % Compute direction
    p = -B * g;

    % Compute step size
    phid0 = g' * p;
    p = reshape(p, [h,m]);
    if strcmp(line_search, 'BLS')
        [a, v_new, g_new] = BacktrackingLS(1, 0.9, 1e-4, beta, p, E, v, phid0, X, T, W, b, N, f, lambda);
    else
        [a, v_new, g_new] = ArmijoWolfeLS(1, 0.9, 1e-4, 0.9, beta, p, E, v, phid0, X, T, W, b, N, f, lambda);
    end
    
    if (v_new - opt_val) <= precision
       prec_tEnd = toc(prec_tStart);
       got_precision = true;
    end

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
    errors(iter+1) = v;
end
tEnd = toc(tStart);

if print_stat
    fprintf('\n### BFGS (%s) ###\n', line_search)
    fprintf('# iterations = %d\nFinal error = %d\nElapsed time/iteration = %d\n', iter, v, tEnd / iter);
    all_decreasing = true;

    % Test all decreasing errors
    for i = 1:iter
        if errors(i) < errors(i+1)
            all_decreasing = false;
            break;
        end
    end
    if all_decreasing
        fprintf('The errors were all decreasing.\n')
    else
        fprintf('The errors were *NOT* all decreasing.\n')
    end
end

end


function [a, phi, g] = BacktrackingLS(a, tau, c1, beta, p, E, phi0, phid0, X, T, W, b, N, f, lambda)
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
    [phi, g] = E(beta + a * p, X, T, W, b, N, f, lambda);
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
