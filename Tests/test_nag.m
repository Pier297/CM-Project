rng(1)

X = zeros(100, 1);
T = zeros(100, 1);
c = 1;
for i = 0:0.1:10
    X(c) = i;
    T(c) = sin(i);
    c = c + 1;
end

f = @tanh;                    % hidden activation function
eps = 1e-1;

n = size(X,2);          % input dimension
m = size(T,2);          % output dimension
N = size(X,1);          % number of samples
h = N;                        % number of hidden units  
W = randn(h,n);         % weight between input and hidden layer
b = randn(h,1);         % bias of hidden nodes
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier

beta = randn(h,m);      % randomly initialized beta

[~, ~, lambda] = grid_search(@NAG, @ObjectiveFunc, X, T, f, eps, N, W, b, beta, [], []);

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

%beta = NAG(@ObjectiveFunc, beta, eps);
[beta_nag, errors] = NAG(@ObjectiveFunc, beta, 1e-3, eta, lambda, N, X, T, W, b, f, true, intmax, intmax);

 % Plot the stats
figure
scatter(1:(length(errors)), errors)
title('NAG | Error function')
xlabel('iteration')
ylabel('Error')

figure
% plot training data
scatter(X, T)
title('NAG | training data vs model prediction')
xlabel('x')
ylabel('sin(x)')
c = 1;
for i = 0:0.1:10
    X(c) = i;
    T(c) = sin(i);
    c = c + 1;
end
hold on
Y = [];
for i = 1:N
   Y = [Y, out(f, W, b, beta_nag, X(:, i))]; 
end
plot(X, Y)
legend({'Training data', 'Model prediction'}, 'Location', 'southwest')

%B = eye(h*m);
%beta_bfgs = BFGS(@ObjectiveFunc, beta, B, eps);

    function [elm_out] = out(f, W, b, beta_nag, x)
        elm_out = beta_nag' * f(W * x + b);
    end