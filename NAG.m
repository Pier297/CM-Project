function [beta, errors] = NAG(E, beta, eps, eta, lambda, alpha_t_minus_1, N, X, T, W, b, f)
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

a_t = 1;
delta_beta_t_minus_1 = 0;

[v,gr] = E(beta, X, T, W, b, N, f, lambda);

errors = [v];
iter = 0;

MAX_ITER = 2000;

while iter < MAX_ITER && norm(gr) > eps
    a_t_plus_1 = update_a(a_t);
    alpha_t = (a_t - 1)/a_t_plus_1;

    [~, g] = E(beta + alpha_t_minus_1 * delta_beta_t_minus_1, X, T, W, b, N, f, lambda);

    delta_beta_t = alpha_t_minus_1 * delta_beta_t_minus_1 - eta * g; %- (2*lambda/N * beta);

    beta = beta + delta_beta_t;

    a_t = a_t_plus_1;
    alpha_t_minus_1 = alpha_t;
    delta_beta_t_minus_1 = delta_beta_t;
    
    [v,gr] = E(beta, X, T, W, b, N, f, lambda);

    errors = [errors, v];
    iter = iter + 1;
end

end