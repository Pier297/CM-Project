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

    delta_beta_t = alpha_t_minus_1 * delta_beta_t_minus_1 - eta * g - 2*lambda/N * beta;

    beta = beta + delta_beta_t;

    a_t = a_t_plus_1;
    alpha_t_minus_1 = alpha_t;
    delta_beta_t_minus_1 = delta_beta_t;
    
    [v,gr] = E(beta);
    %fprintf('%d\t%d\n', v, norm(gr));
    errors = [errors, v];
    iter = iter + 1;
end

fprintf('\n### NAG ###\n')
fprintf('\n# iterations = %d\n\nFinal error = %d\n\n', iter, v);

figure
scatter(1:iter+1, errors)
title('NAG | Error function')
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

figure
% plot training data
scatter(X, T)
title('NAG | training data vs model prediction')
xlabel('x')
ylabel('sin(x)')
c = 1;
for i = 0:0.1:10
    X(c) = i;
    T(c) = sin(i);
    c = c + 1;
end
hold on
Y = [];
for i = 1:N
   Y = [Y, out(X(:, i))]; 
end
plot(X, Y)
legend({'Training data', 'Model prediction'}, 'Location', 'southwest')
end