rng(1);                 % seed to make random values repeatable

X = zeros(100, 1);
T = zeros(100, 1);
c = 1;
for i = 0:0.1:10
    X(c) = i;
    T(c) = sin(i); %+ (rand - 0.5)/3;
    c = c + 1;
end

scatter(X, T)
xlabel('x', 'FontSize', 14)
ylabel('sin(x)', 'FontSize', 14)
saveas(gcf, 'Plots/sin_training_data.png')

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

%[beta_nag, errors, lambda] = grid_search(@NAG, @ObjectiveFunc, X, T, f, eps, N, W, b, beta);
lambda = 0;

[beta_opt, opt_val, opt_val_grad] = true_solution(X, T, W, b, f, N, h, m, lambda);
fprintf('MSE = %d\n', opt_val);

hessian = 0;
for i = 1:N
    x = X(:,i);
    t = T(:,i);
    hidden_out = f(W * x + b);
    hessian = hessian + (hidden_out * hidden_out');
end
hessian = 2/N * (hessian + lambda);
eta = 1/norm(hessian);
[beta_nag, errors_nag] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, true, intmax, 0);
fprintf('MSE = %d\n', errors_nag(length(errors_nag)));

figure
% plot training data
scatter(X, T)
title('NAG | training data vs model prediction')
xlabel('x', 'FontSize', 14)
ylabel('sin(x)', 'FontSize', 14)
hold on
Y = [];
for i = 1:N
   Y = [Y, out(f, W, b, beta_nag, X(:, i))]; 
end
plot(X, Y)
saveas(gcf, 'Plots/NAG_sin_prediction_vs_training_data.png')
legend({'Training data', 'Model prediction'}, 'Location', 'southwest')


% ------- BFGS (BLS) -------
B = eye(h*m);
[beta_bfgs_bls, errors_bfgs_bls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS');
fprintf('MSE = %d\n', errors_bfgs_bls(length(errors_bfgs_bls)));

figure
% plot training data
scatter(X, T)
title('BFGS (BLS) | training data vs model prediction')
xlabel('x', 'FontSize', 14)
ylabel('sin(x)', 'FontSize', 14)
hold on
Y = [];
for i = 1:N
   Y = [Y, out(f, W, b, beta_bfgs_bls, X(:, i))]; 
end
plot(X, Y)
saveas(gcf, 'Plots/BFGS_BLS_sin_prediction_vs_training_data.png')
legend({'Training data', 'Model prediction'}, 'Location', 'southwest')


% ------- BFGS (AWLS) -------
B = eye(h*m);
[beta_bfgs_awls, errors_bfgs_awls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS');
fprintf('MSE = %d\n', errors_bfgs_awls(length(errors_bfgs_awls)));

figure
% plot training data
scatter(X, T)
title('BFGS (AWLS) | training data vs model prediction')
xlabel('x', 'FontSize', 14)
ylabel('sin(x)', 'FontSize', 14)
hold on
Y = [];
for i = 1:N
   Y = [Y, out(f, W, b, beta_bfgs_awls, X(:, i))]; 
end
plot(X, Y)
saveas(gcf, 'Plots/BFGS_AWLS_sin_prediction_vs_training_data.png')
legend({'Training data', 'Model prediction'}, 'Location', 'southwest')

% ---
% Plot log convergence
errors_nag = errors_nag - opt_val;
errors_bfgs_bls = errors_bfgs_bls - opt_val;
errors_bfgs_awls = errors_bfgs_awls - opt_val;


figure
semilogy(1:(length(errors_nag)), errors_nag)
title('NAG')
xlabel('iteration', 'FontSize', 14)
ylabel('log(Error)', 'FontSize', 14)
saveas(gcf, 'Plots/sin_NAG_convergence_rate.png')

figure
semilogy(1:(length(errors_bfgs_awls)), errors_bfgs_awls)
title('BFGS (AWLS)')
xlabel('iteration', 'FontSize', 14)
ylabel('log(Error)', 'FontSize', 14)
saveas(gcf, 'Plots/sin_BFGS_AWLS_convergence_rate.png')


figure
semilogy(1:(length(errors_bfgs_bls)), errors_bfgs_bls)
title('BFGS (BLS)')
xlabel('iteration', 'FontSize', 14)
ylabel('log(Error)', 'FontSize', 14)
saveas(gcf, 'Plots/sin_BFGS_BLS_convergence_rate.png')

%, 1:(length(errors_nag)), errors_nag, 1:(length(errors_bfgs_bls)), errors_bfgs_bls, 1:(length(errors_bfgs_awls)), errors_bfgs_awls)
%legend('GD', 'NAG', 'BFGS (BLS)', 'BFGS (AWLS)')



    function [elm_out] = out(f, W, b, beta, x)
        elm_out = beta' * f(W * x + b);
    end