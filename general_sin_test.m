

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
eps = 1e-4;

n = size(X,2);          % input dimension
m = size(T,2);          % output dimension
N = size(X,1);          % number of samples
h = N;                        % number of hidden units  
W = randn(h,n);         % weight between input and hidden layer
b = randn(h,1);         % bias of hidden nodes
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier

beta = randn(h,m);      % randomly initialized beta

[alpha_t_minus_1, lambda] = grid_search(@NAG, @ObjectiveFunc, X, T, f);

fprintf('\n### NAG ###\n')

fprintf('Finish grid search, found:\n alpha = %d\n lambda=%d\n', alpha_t_minus_1, lambda)

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

[beta_nag, errors] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, alpha_t_minus_1, N, X, T, W, b, f);


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

% ------- BFGS ------

B = eye(h*m);
[beta_bfgs, errors_bfgs] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N);

figure
% plot training data
scatter(X, T)
title('BFGS | training data vs model prediction')
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
   Y = [Y, out(f, W, b, beta_bfgs, X(:, i))]; 
end
plot(X, Y)
legend({'Training data', 'Model prediction'}, 'Location', 'southwest')


    function [elm_out] = out(f, W, b, beta, x)
        elm_out = beta' * f(W * x + b);
    end