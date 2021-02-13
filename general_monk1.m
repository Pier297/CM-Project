% --- parameter
filename = 'monk1-train.txt';
f = @tanh;              % hidden activation function
h = 124;                % number of hidden units
lambda = 0;             % regularization parameter, obtained from grid search
alpha = 0.9;            % momentum coefficient, obtained from grid search
eps = 1e-4;
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


% ------- True Solution -------
[beta_opt, opt_val, opt_val_grad] = true_solution(X, T, W, b, f, N, h, m, lambda);


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
[beta_nag, errs] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, alpha, N, X, T, W, b, f, true, intmax);
fprintf('Accuracy = %d\n', accuracy(X, T, W, b, f, N, beta_nag));


% ------- BFGS (BLS) -------
B = eye(h*m);
[beta_bfgs_bls, errors_bfgs_bls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS');
fprintf('Accuracy = %d\n', accuracy(X, T, W, b, f, N, beta_bfgs_bls));


% ------- BFGS (AWLS) -------
B = eye(h*m);
[beta_bfgs_awls, errors_bfgs_awls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS');
fprintf('Accuracy = %d\n', accuracy(X, T, W, b, f, N, beta_bfgs_awls));





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