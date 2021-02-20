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
eps = 1e-6;

n = size(X,2);          % input dimension
m = size(T,2);          % output dimension
N = size(X,1);          % number of samples
h = N;                        % number of hidden units  
W = randn(h,n);         % weight between input and hidden layer
b = randn(h,1);         % bias of hidden nodes
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier

beta = randn(h,m);      % randomly initialized beta

%[beta_nag, errors, lambda] = grid_search(@NAG, @ObjectiveFunc, X, T, f, eps, N, W, b, beta, [], []);
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
eta = 1/norm(hessian)

[beta_nag, errors_nag] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, true, intmax, 0);
fprintf('MSE = %d\n', errors_nag(length(errors_nag)));


% ------- BFGS (BLS) -------
B = eye(h*m);
[beta_bfgs_bls, errors_bfgs_bls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', true);
fprintf('MSE = %d\n', errors_bfgs_bls(length(errors_bfgs_bls)));

% ------- BFGS (AWLS) -------
B = eye(h*m);
[beta_bfgs_awls, errors_bfgs_awls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS', true);
fprintf('MSE = %d\n', errors_bfgs_awls(length(errors_bfgs_awls)));


% ------------- PLOT -----------

% Real solution
figure
scatter(X, T)
hold on
Y = [];
for i = 1:N
   Y = [Y, out(f, W, b, beta_opt, X(:, i))]; 
end
plot(X, Y)

% NAG
hold on
Y = [];
for i = 1:N
   Y = [Y, out(f, W, b, beta_nag, X(:, i))]; 
end
plot(X, Y)

% BFGS (BLS)
hold on
Y = [];
for i = 1:N
   Y = [Y, out(f, W, b, beta_bfgs_bls, X(:, i))]; 
end
plot(X, Y)

% BFGS (AWLS)
hold on
Y = [];
for i = 1:N
   Y = [Y, out(f, W, b, beta_bfgs_awls, X(:, i))]; 
end
plot(X, Y)

xlabel('x', 'FontSize', 14)
ylabel('sin(x)', 'FontSize', 14)
legend({'Training data', 'Model prediction'}, 'Location', 'southwest')
legend('Training Data', 'Real Solution', 'NAG', 'BFGS (BLS)', 'BFGS (AWLS)')

%saveas(gcf, 'Plots/e10-1_h100_sin_prediction_vs_training_data.png')
%saveas(gcf, 'Plots/e10-3_h100_sin_prediction_vs_training_data.png')
saveas(gcf, 'Plots/e10-6_h100_sin_prediction_vs_training_data.png')

% --- Plot log convergence
errors_nag = errors_nag - opt_val;
errors_bfgs_bls = errors_bfgs_bls - opt_val;
errors_bfgs_awls = errors_bfgs_awls - opt_val;


figure
semilogy(1:(length(errors_nag)), errors_nag, 1:(length(errors_bfgs_awls)), errors_bfgs_awls, 1:(length(errors_bfgs_bls)), errors_bfgs_bls)
xlabel('iteration', 'FontSize', 14)
ylabel('log(Error)', 'FontSize', 14)
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)')

%saveas(gcf, 'Plots/10-1_100_sin_convergence_rate.png')
%saveas(gcf, 'Plots/10-3_100_sin_convergence_rate.png')
saveas(gcf, 'Plots/10-6_100_sin_convergence_rate.png')

figure
semilogy(1:(length(errors_bfgs_awls)), errors_bfgs_awls, 1:(length(errors_bfgs_bls)), errors_bfgs_bls)
xlabel('iteration', 'FontSize', 14)
ylabel('log(Error)', 'FontSize', 14)
legend('BFGS (BLS)', 'BFGS (AWLS)')

%saveas(gcf, 'Plots/10-3_100_sin_BFGS_only_convergence_rate.png')
saveas(gcf, 'Plots/10-6_100_sin_BFGS_only_convergence_rate.png')

    function [elm_out] = out(f, W, b, beta, x)
        elm_out = beta' * f(W * x + b);
    end