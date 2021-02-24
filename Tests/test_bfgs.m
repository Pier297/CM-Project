rng(1)

X = zeros(100, 1);
T = zeros(100, 1);
c = 1;
for i = 0:0.1:10
    X(c) = i;
    T(c) = sin(i);
    c = c + 1;
end
lambda = 0.001;               % regularization parameter
f = @tanh;                    % hidden activation function
eps = 1e-1;

n = size(X,2);          % input dimension
m = size(T,2);          % output dimension
N = size(X,1);          % number of samples
h = N;
W = randn(h,n);         % weight between input and hidden layer
b = randn(h,1);         % bias of hidden nodes
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier

beta = randn(h,m);      % randomly initialized beta

B = eye(h*m);           % initial approximation of inverse Hessian

beta = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', true);
