global X
global T
global h
global eta
global lambda
global alpha_t_minus_1
global f
global W
global b
global m
global N

%X = [1,2,3 ; 2,3,4 ; 3,4,5];  % input
%T = [2,5 ; 4,6 ; 6,8];        % target
X = zeros(100, 1);
T = zeros(100, 1);
c = 1;
for i = 0:0.1:10
    X(c) = i;
    T(c) = sin(i);
    c = c + 1;
end

f = @tanh;                    % hidden activation function
eps = 1e-3;

n = size(X,2);          % input dimension
m = size(T,2);          % output dimension
N = size(X,1);          % number of samples
h = N;                        % number of hidden units  
W = randn(h,n);         % weight between input and hidden layer
b = randn(h,1);         % bias of hidden nodes
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier

beta = randn(h,m);      % randomly initialized beta

lambda = 0.000001; % regularization parameter

alpha_t_minus_1 = 0.0001; % momentum constant

% Compute hessian
hessian = 0;
for i = 1:N
    x = X(:,i);
    t = T(:,i);
    hidden_out = f(W * x + b);
    hessian = hessian + (hidden_out * hidden_out');
end

hessian = 2/N * (hessian + lambda);

eta = 1/norm(hessian);

beta = NAG(@ObjectiveFunc, beta, eps);