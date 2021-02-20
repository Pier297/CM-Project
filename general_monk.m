% --- parameter
filename = 'data/monk1-train.txt';
f = @tanh;              % hidden activation function
h = 124;                % number of hidden units
eps = 1e-6;
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

%[beta_nag, errors, lambda] = grid_search(@NAG, @ObjectiveFunc, X, T, f, eps, N, W, b, beta, [], []);
lambda = 0;

% ------- True Solution -------
[beta_opt, opt_val, opt_val_grad] = true_solution(X, T, W, b, f, N, h, m, lambda);
fprintf('Optimal value = %d\n', opt_val);
fprintf('Accuracy = %d\n', accuracy(X, T, W, b, f, N, beta_opt));


% ------- Normal Equation -------
beta_neq = normal_equation(X', T', W, b, N, h, f);
fprintf('Accuracy = %d\n', accuracy(X, T, W, b, f, N, beta_neq));


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
[beta_nag, errors_nag] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, true, 5000, 0);
fprintf('Accuracy = %d\n', accuracy(X, T, W, b, f, N, beta_nag));


% ------- BFGS (BLS) -------
B = eye(h*m);
[beta_bfgs_bls, errors_bfgs_bls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', true);
fprintf('Accuracy = %d\n', accuracy(X, T, W, b, f, N, beta_bfgs_bls));


% ------- BFGS (AWLS) -------
B = eye(h*m);
[beta_bfgs_awls, errors_bfgs_awls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS', true);
fprintf('Accuracy = %d\n', accuracy(X, T, W, b, f, N, beta_bfgs_awls));


% ------- Plot log scale -------
errors_nag = errors_nag - opt_val;
errors_bfgs_bls = errors_bfgs_bls - opt_val;
errors_bfgs_awls = errors_bfgs_awls - opt_val;


figure
semilogy(1:(length(errors_nag)), errors_nag)
title('NAG')
xlabel('iteration', 'FontSize', 14)
ylabel('log(Error)', 'FontSize', 14)
%saveas(gcf, 'Plots/monk_NAG_convergence_rate.png')

figure
semilogy(1:(length(errors_bfgs_awls)), errors_bfgs_awls)
title('BFGS (AWLS)')
xlabel('iteration', 'FontSize', 14)
ylabel('log(Error)', 'FontSize', 14)
%saveas(gcf, 'Plots/monk_BFGS_AWLS_convergence_rate.png')


figure
semilogy(1:(length(errors_bfgs_bls)), errors_bfgs_bls)
title('BFGS (BLS)')
xlabel('iteration', 'FontSize', 14)
ylabel('log(Error)', 'FontSize', 14)
%saveas(gcf, 'Plots/monk_BFGS_BLS_convergence_rate.png')

figure
semilogy(1:(length(errors_nag)), errors_nag, 1:(length(errors_bfgs_bls)), errors_bfgs_bls, 1:(length(errors_bfgs_awls)), errors_bfgs_awls)
title('Convergence')
xlabel('iteration', 'FontSize', 14)
ylabel('log(Error)', 'FontSize', 14)
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)')
%saveas(gcf, 'Plots/monk_convergence_rate.png')



function score = accuracy(X, T, W, b, f, N, beta)
    correct = 0;
    for i = 1:N
        prediction = beta' * f(W * X(:, i) + b);
        if prediction > 0.5
            if T(i) == 1
                correct = correct + 1;
            end
        else
            if T(i) == 0
                correct = correct + 1;
            end
        end
    end
    score = correct/N;
end