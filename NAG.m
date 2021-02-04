function [beta] = NAG(E, beta, eps)
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

global h
global n
global m
global N
global T
global W
global X
global b
global f
global eta % Learning rate
global lambda % regularization parameter, found by grid search
global alpha_t_minus_1 % momentum constant, found by grid search


    function [a_t_plus_1] = update_a(a_t)
       a_t_plus_1 = (1 + sqrt(4 * a_t^2 + 1))/2;
    end

    function [elm_out] = out(x)
        elm_out = beta' * f(W * x + b);
    end

a_t = 1;
a_t_plus_1 = update_a(a_t);
delta_beta_t_minus_1 = 0;

[v,gr] = E(beta);
errors = [v];
iter = 0;

while norm(gr) > eps
    a_t_plus_1 = update_a(a_t);
    alpha_t = (a_t - 1)/a_t_plus_1;

    [~, g] = E(beta + alpha_t_minus_1 * delta_beta_t_minus_1);

    delta_beta_t = alpha_t_minus_1 * delta_beta_t_minus_1 - eta * g;

    beta = beta + delta_beta_t;

    a_t = a_t_plus_1;
    alpha_t_minus_1 = alpha_t;
    delta_beta_t_minus_1 = delta_beta_t;
    
    [v,gr] = E(beta);
    errors = [errors, v];
    iter = iter + 1;
end

scatter(1:iter+1, errors)
iter, v
all_decreasing = true;
% Test all decreasing errors
for i = 1:iter
    if errors(i) < errors(i+1)
        all_decreasing = false;
        break;
    end
end
all_decreasing

end