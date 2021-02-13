rng(1);                 % seed to make random values repeatable

X = zeros(100, 1);
T = zeros(100, 1);
c = 1;
for i = 0:0.1:10
    X(c) = i;
    T(c) = sin(i) + (rand - 0.5)/3;
    c = c + 1;
end

f = @tanh;                    % hidden activation function
eps = 1e-4;

n = size(X,2);          % input dimension
m = size(T,2);          % output dimension
N = size(X,1);          % number of samples
h = N;                        % number of hidden units  
W = randn(h,n);         % weight between input and hidden layer
b = randn(h,1);         % bias of hidden nodes
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier
% lambda = 0.0001

beta = randn(h,m);      % randomly initialized beta

[beta_nag, errors, alpha, lambda] = grid_search(@NAG, @ObjectiveFunc, X, T, f, eps, N, W, b, beta);

fprintf('\n### NAG ###\n')
fprintf('%d iterations\n', length(errors))
fprintf('Final error = %d\n', errors(length(errors)))

all_decreasing = true;
for i = 1:(length(errors)-1)
    if errors(i) < errors(i+1)
        all_decreasing = false;
        break;
    end
end
if all_decreasing
    fprintf('The errors were all decreasing.\n\n')
else
    fprintf('The errors were *NOT* all decreasing.\n')
end

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
hold on
Y = [];
for i = 1:N
   Y = [Y, out(f, W, b, beta_nag, X(:, i))]; 
end
plot(X, Y)
legend({'Training data', 'Model prediction'}, 'Location', 'southwest')

% ------- BFGS ------

B = eye(h*m);
[beta_bfgs, errors_bfgs] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS');

figure
% plot training data
scatter(X, T)
title('BFGS | training data vs model prediction')
xlabel('x')
ylabel('sin(x)')
hold on
Y = [];
for i = 1:N
   Y = [Y, out(f, W, b, beta_bfgs, X(:, i))]; 
end
plot(X, Y)
legend({'Training data', 'Model prediction'}, 'Location', 'southwest')


    function [elm_out] = out(f, W, b, beta, x)
        elm_out = beta' * f(W * x + b);
    end