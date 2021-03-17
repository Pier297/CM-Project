rng(1);

% --- parameter
f = @tanh;              % hidden activation function
n = 5;
eps = 1e-3;

precision = 1e-5;

N = 250;
h = 250;                % number of hidden units
m = 5;
% --- end of parameter

X = randn(N, n);
T = randn(N, m);

W = rand(h,n)*2-1;      % weight between input and hidden layer, range in [-1,1]
b = rand(h,1)*2-1;      % bias of hidden nodes, range in [-1,1]
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier
beta = rand(h,m)*2-1;   % randomly initialized beta, range in [-1,1]

lambda = 0;

% ------- True Solution -------
[beta_opt, opt_val, opt_val_grad] = true_solution(X, T, W, b, f, N, h, m, lambda);
fprintf('MSE = %d\n', opt_val);

% ------- NAG -------
hessian = 0;
for i = 1:N
    x = X(:,i);
    t = T(:,i);
    hidden_out = f(W * x + b);
    hessian = hessian + (hidden_out * hidden_out');
end
hessian = 2/N * (hessian + lambda);
eta = 1/norm(hessian);
[beta_nag, errors_nag, ~, ~] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, true, intmax, intmax, opt_val, precision, true);


B = eye(h*m);
[beta_bfgs_bls, errors_bfgs_bls, ~, bls_prec_tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', true, opt_val, precision, true);


B = eye(h*m);
[beta_bfgs_awls, errors_bfgs_awls, ~, awls_prec_tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS', true, opt_val, precision, true);

% ------- Plot log scale -------
figure
semilogy(1:(length(errors_nag)), errors_nag, 1:(length(errors_bfgs_awls)), errors_bfgs_awls, 1:(length(errors_bfgs_bls)), errors_bfgs_bls)
xlabel('iteration', 'FontSize', 14)
ylabel('( E(\beta) - E* ) / E*', 'FontSize', 14)
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)')
saveas(gcf, 'Plots/random_convergence_rate_t1.png')
