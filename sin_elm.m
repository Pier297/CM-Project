% w = matrix of weights from input to hidden layer, R^h x n
%     each row i has the weights connected to the inputs; So,
%     w[1] contains a list of n weights.
% b[i] is the bias of hidden unit i, b belongs in R^h
% H is a matrix where each row corresponds to the output of the hidden
%   layer on input x[j]. H[i] = tanh(w[i]*x[j] + b[i]) 
clf
x = zeros(100, 1);
t = zeros(100, 1);
c = 1;
for i = 0:0.1:10
    x(c) = i;
    t(c) = sin(i);
    c = c + 1;
end

% n is input dimension
n = size(x); n = n(1,2);
% m is output dimension
m = size(t); m = m(1,2);
% N is number of samples
N = size(x); N = N(1,1);
T = t;

% h is number of hidden units
h = N;

% Step 1
w = randn(h, n);
b = randn(h, 1);

% Step 2
H = zeros(N, h);
for j = 1:N
    for i = 1:h
        H(j, i) = tanh(transpose(w(i)) * transpose(x(j)) + b(i));
    end
end

% Step 3
B = pinv(H) * T;

norm(H*B - T)
plot(x,H*B)
hold on
scatter(x,t)