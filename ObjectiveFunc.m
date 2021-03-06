function [v, g] = ObjectiveFunc(beta, X, T, W, b, N, f, lambda)
% Mean-Squared Error with L2 regularization
%
% Output:
%   v = E(\beta)
%   g = \nabla E(\beta)

mse = 0;
nabla = 0;
for i = 1:N
    x = X(:,i);
    t = T(:,i);
    hidden_out = f(W * x + b); % output from hidden layer
    e = beta' * hidden_out - t;
    mse = mse + (e' * e);
    nabla = nabla + (hidden_out * e');
end

v = (mse + (lambda * norm(beta,'fro')^2)) / N;
g = 2 * (nabla + lambda * beta) / N;

end
