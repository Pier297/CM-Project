function [beta, errors] = NAG(E, beta, eps, eta, lambda, N, X, T, W, b, f, print_stat, MAX_ITER, MAX_UNLUCKY_STEPS)
% Nesterov' Accelerated Gradient Descent with L2 regularization
% Inputs:
%   E:    Error function
%         E(beta) returns [v, g], where v is the value
%         and g is the gradient
%
%   beta: Initial weights
%
%   eps:  Stopping criteria, it indicates that when the norm
%         of the gradient is equal to 'eps' then we assume it's 0
%         or equally that we are at a minima.
%
% Outputs:
%   beta: Final weights

    function [a_t_plus_1] = update_a(a_t)
       a_t_plus_1 = (1 + sqrt(4 * a_t^2 + 1))/2;
    end

a_t_minus_1 = 1;
delta_beta_t_minus_1 = 0;

[v,gr] = E(beta, X, T, W, b, N, f, lambda);

errors = (v);
iter = 0;

prevError = v;
unluckySteps = 0;

tStart = tic;
while unluckySteps <= MAX_UNLUCKY_STEPS && iter < MAX_ITER && norm(gr) > eps
    a_t = update_a(a_t_minus_1);
    
    alpha_t_minus_1 = (a_t_minus_1 - 1)/a_t;

    [~, g] = E(beta + alpha_t_minus_1 * delta_beta_t_minus_1, X, T, W, b, N, f, lambda);

    delta_beta_t = alpha_t_minus_1 * delta_beta_t_minus_1 - eta * g;

    beta = beta + delta_beta_t;
    
    a_t_minus_1 = a_t;
    delta_beta_t_minus_1 = delta_beta_t;
    
    [v,gr] = E(beta, X, T, W, b, N, f, lambda);
    
    if v >= prevError
        unluckySteps = unluckySteps + 1;
    else
        unluckySteps = 0;
    end
    
    prevError = v;

    iter = iter + 1;
    errors(iter) = v;
end
tEnd = toc(tStart);

if print_stat
    fprintf('\n### NAG ###\n')
    fprintf('# iterations = %d\nFinal error = %d\nElapsed time = %d\n', iter, v, tEnd);
    
    % Test all decreasing errors
    all_decreasing = true;
    for i = 1:iter-1
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