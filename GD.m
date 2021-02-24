function [beta, errors] = GD(E, beta, eps, h, m, N, T, W, X, b, f, eta, lambda)
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

[v, g] = E(beta, X, T, W, b, N, f, lambda);

    function [elm_out] = out(x)
        elm_out = beta' * f(W * x + b);
    end

iter = 0;
errors = (v);

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
    [v, g] = E(beta, X, T, W, b, N, f, lambda);
    iter = iter + 1;
    errors(iter) = v;
end

all_decreasing = true;
% Test all decreasing errors
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
fprintf('MSE = %d\n', v)

end