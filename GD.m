function [beta] = GD(E, beta, eps)
% Gradient Descent with L2 regularization
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
global eta
global lambda


[v, g] = E(beta);

    function [elm_out] = out(x)
        elm_out = beta' * f(W * x + b);
    end

iter = 0;
errors = [v];
v
while (norm(g) > eps)
    % Compute the summation
    r = zeros(h, m);
    
    for i = 1:N
       for j = 1:h
           x_i = X(:, i);
           t_i = T(:, i);
           w_j = W(j, :); % row vector of 1x2
           
           % r(j, :) = row vector of 1x2
           
           diff = out(x_i) - t_i; % 2x1
           hidden_out = f(w_j * x_i + b(j)); % 1x1
           r(j, :) = r(j, :) + (hidden_out * diff)';
       end
    end
    
    % Update beta
    delta_beta = (-eta * 2/N * r) - (2*lambda/N * beta);
    beta = beta + delta_beta;
    [v, g] = E(beta);
    v;
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