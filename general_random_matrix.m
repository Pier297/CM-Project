% --- parameter
f = @tanh;              % hidden activation function
h = 100;                % number of hidden units
lambda = 0;             % regularization parameter, obtained from grid search
alpha = 0.9;            % momentum coefficient, obtained from grid search
eps = 1e-1;
% --- end of parameter

X = randn(5000, 10);
T = randn(5000, 5);

rng(1);                 % seed to make random values repeatable
n = size(X,2);          % input dimension
m = size(T,2);          % output dimension
N = size(X,1);          % number of samples
W = rand(h,n)*2-1;      % weight between input and hidden layer, range in [-1,1]
b = rand(h,1)*2-1;      % bias of hidden nodes, range in [-1,1]
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier
beta = rand(h,m)*2-1;   % randomly initialized beta, range in [-1,1]


% ------- True Solution -------
[beta_opt, opt_val, opt_val_grad] = true_solution(X, T, W, b, f, N, h, m, lambda);
fprintf('MSE = %d\n', opt_val);


% ------- Normal Equation -------
beta_neq = normal_equation(X', T', W, b, N, h, f);
[v,~] = ObjectiveFunc(beta_neq, X, T, W, b, N, f, lambda);
fprintf('MSE = %d\n', v);


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
[beta_nag, errs] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, alpha, N, X, T, W, b, f, true, 280, 0);
fprintf('MSE = %d\n', errs(length(errs)));


% ------- BFGS (BLS) -------
B = eye(h*m);
[beta_bfgs_bls, errors_bfgs_bls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS');
fprintf('MSE = %d\n', errors_bfgs_bls(length(errors_bfgs_bls)));


% ------- BFGS (AWLS) -------
B = eye(h*m);
[beta_bfgs_awls, errors_bfgs_awls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS');
fprintf('MSE = %d\n', errors_bfgs_awls(length(errors_bfgs_awls)));


% ------- Plot log scale -------
errs = errs - opt_val;
errors_bfgs_bls = errors_bfgs_bls - opt_val;
errors_bfgs_awls = errors_bfgs_awls - opt_val;

figure
semilogy(1:(length(errs)), errs, 1:(length(errors_bfgs_bls)), errors_bfgs_bls, 1:(length(errors_bfgs_awls)), errors_bfgs_awls)
title('??')
xlabel('iteration')
ylabel('log(Error)')
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)')
