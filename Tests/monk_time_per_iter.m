% --- parameter
filename = 'data/monk2-train.txt';
f = @tanh;              % hidden activation function
h = 169;                % number of hidden units
k = 10;                 % how many times to repeat the calculation
eps = 1e-6;
lambda = 0;
% --- end of parameter


input = load(filename);
[row, cols] = size(input);
X = input(1:row, 1:cols-1);
T = input(1:row, cols:cols);

rng(1);                 % seed to make random values repeatable
n = size(X,2);          % input dimension
m = size(T,2);          % output dimension
N = size(X,1);          % number of samples
W = rand(h,n)*2-1;      % weight between input and hidden layer, range in [-1,1]
b = rand(h,1)*2-1;      % bias of hidden nodes, range in [-1,1]
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier
beta = rand(h,m)*2-1;   % randomly initialized beta, range in [-1,1]

nag_times = (0);
bfgs_awls_times = (0);
bfgs_bls_times = (0);
nag_iters = (0);
bfgs_awls_iters = (0);
bfgs_bls_iters = (0);


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
for i = 1:k
    [~, ~, iter, tEnd] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, false, 5000, 0);
    nag_times(i) = tEnd;
    nag_iters(i) = iter;
end


% ------- BFGS (BLS) -------
B = eye(h*m);
for i = 1:k
    [~, ~, iter, tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', false);
    bfgs_bls_times(i) = tEnd;
    bfgs_bls_iters(i) = iter;
end


% ------- BFGS (AWLS) -------
B = eye(h*m);
for i = 1:k
    [~, ~, iter, tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS', false);
    bfgs_awls_times(i) = tEnd;
    bfgs_awls_iters(i) = iter;
end


fprintf('Average time per iteration\n')
fprintf('NAG = %d\n', mean(nag_times./nag_iters));
fprintf('BFGS (BLS) = %d\n', mean(bfgs_bls_times./bfgs_bls_iters));
fprintf('BFGS (AWLS) = %d\n', mean(bfgs_awls_times./bfgs_awls_iters));
