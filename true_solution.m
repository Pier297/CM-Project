function [beta, v, g] = true_solution(X, T, W, b, f, N, h, m, lambda)

tStart = tic;

% --- compute 1st term: \sum_{i=1}^{N} f(W x_i + b) f(W x_i + b)^T + \lambda I
first_term = zeros(h,h);
for i = 1:N
    a = f(W * X(:,i) + b);
    first_term = first_term + a * a';
end
first_term = first_term + lambda * eye(h,h);

% --- compute 2nd term: \sum_{i=1}^{N} f(W x_i + b) t_i^T
second_term = zeros(h,m);
for i = 1:N
    a = f(W * X(:,i) + b);
    second_term = second_term + a * T(:,i)';
end

% --- final result
beta = first_term \ second_term;

tEnd = toc(tStart);

fprintf('\n### True Solution ###\n');
fprintf('Time elapsed = %d\n', tEnd);

% --- verify
[v, g] = ObjectiveFunc(beta, X, T, W, b, N, f, lambda);


end

