global X
global T
global h
global eta
global lambda
global f
global W
global b
global m
global N

X = [1,2,3 ; 2,3,4 ; 3,4,5];  % input
T = [2,5 ; 4,6 ; 6,8];        % target
h = 4;                        % number of hidden units          
f = @tanh;                    % hidden activation function
eps = 1e-2;

n = size(X,2);          % input dimension
m = size(T,2);          % output dimension
N = size(X,1);          % number of samples
W = randn(h,n);         % weight between input and hidden layer
b = randn(h,1);         % bias of hidden nodes
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier

beta = randn(h,m);      % randomly initialized beta

lambda = 0.00001; % regularization parameter

%hessian = zeros();

%eta = 1/norm(hessian);
eta = 0.1;

beta = GD(@ObjectiveFunc, beta, eps);