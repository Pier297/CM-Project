rng(1);   

% --- parameter
f = @tanh;              % hidden activation function
n = 5;
eps = 1e-3;

N = 250;
h = N;                % number of hidden units
m = 5;
% --- end of parameter

X = randn(N, n);
T = randn(N, m);

W = rand(h,n)*2-1;      % weight between input and hidden layer, range in [-1,1]
b = rand(h,1)*2-1;      % bias of hidden nodes, range in [-1,1]
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier
beta = rand(h,m)*2-1;   % randomly initialized beta, range in [-1,1]

original_W = W;

[U, S, V] = svd(W);

W_well_conditioned = U * S * V';

%S(1, 1) = 0;
%S(2, 2) = 0;
%S(3, 3) = 0;
S(4, 4) = 0;
S(5, 5) = 0;
b(5) = 0;
%b(1) = 0;
%b(2) = 0;
%b(3) = 0;
b(4) = 0;

W_ill_conditioned = U * S * V';

W = W_ill_conditioned;

%rank(W_well_conditioned)

%rank(W_ill_conditioned)

%norm(beta_well_conditioned - beta)

%norm(beta_ill_conditioned - beta)

%return

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
eta = 1/norm(hessian)
[beta_nag, errors_nag] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, true, intmax, intmax);


% ------- BFGS (BLS) -------
B = eye(h*m);
[beta_bfgs_bls, errors_bfgs_bls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', true);
%fprintf('MSE = %d\n', errors_bfgs_bls(length(errors_bfgs_bls)));


% ------- BFGS (AWLS) -------
B = eye(h*m);
[beta_bfgs_awls, errors_bfgs_awls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS', true);
%fprintf('MSE = %d\n', errors_bfgs_awls(length(errors_bfgs_awls)));


% ------- Plot log scale -------
figure
semilogy(1:(length(errors_nag)), errors_nag, 1:(length(errors_bfgs_awls)), errors_bfgs_awls, 1:(length(errors_bfgs_bls)), errors_bfgs_bls)
xlabel('iteration', 'FontSize', 14)
ylabel('log(Error)', 'FontSize', 14)
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)')
saveas(gcf, 'Plots/random_ill_convergence_rate_t1.png')
