% --- parameter
filename = 'data/monk3-train.txt';
f = @tanh;              % hidden activation function
eps = 1e-3;
lambda = 0.01;
k = 2;
h_min = 0;
h_max = 3000;
step = 300;
% --- end of parameter


input = load(filename);
[row, cols] = size(input);
X = input(1:row, 1:cols-1);
T = input(1:row, cols:cols);

rng(1);                 % seed to make random values repeatable
n = size(X,2);          % input dimension
m = size(T,2);          % output dimension
N = size(X,1);          % number of samples
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier

nag_times = (0);
bfgs_awls_times = (0);
bfgs_bls_times = (0);

iter = 1;

for h = h_min:step:h_max
    W = rand(h,n)*2-1;      % weight between input and hidden layer, range in [-1,1]
    b = rand(h,1)*2-1;      % bias of hidden nodes, range in [-1,1]
    beta = rand(h,m)*2-1;   % randomly initialized beta, range in [-1,1]

    [~, g0] = ObjectiveFunc(beta, X, T, W, b, N, f, lambda);
    ng0 = norm(g0);

    hessian = 0;
    for i = 1:N
        x = X(:,i);
        t = T(:,i);
        hidden_out = f(W * x + b);
        hessian = hessian + (hidden_out * hidden_out');
    end
    hessian = 2/N * (hessian + lambda);
    eta = 1/norm(hessian);
    
    % --- Time NAG
    ticStart = tic;
    for i = 1:k
        [beta_nag, errors_nag] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, false, intmax, intmax);
    end
    tEnd = toc(ticStart);

    nag_times(iter) = (tEnd / k); % / length(errors_nag);
    
    % --- Time BFGS (BLS)
    B = eye(h*m);
    ticStart = tic;
    for i = 1:k
        [beta_bfgs_bls, errors_bfgs_bls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', false);
    end
    tEnd = toc(ticStart);
    
    bfgs_bls_times(iter) = (tEnd / k); % / length(errors_bfgs_bls);
    
    % --- Time BFGS (AWLS)
    B = eye(h*m);
    ticStart = tic;
    for i = 1:k
        [beta_bfgs_awls, errors_bfgs_awls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS', false);
    end
    tEnd = toc(ticStart);

    bfgs_awls_times(iter) = (tEnd / k); % / length(errors_bfgs_awls);

    
    fprintf('%d/%d\n', step*(iter-1), h_max-h_min)
    iter = iter + 1;
end

plot(h_min:step:h_max, nag_times, h_min:step:h_max, bfgs_bls_times, h_min:step:h_max, bfgs_awls_times)
xlabel('Number of hidden nodes', 'FontSize', 14)
ylabel('Computation time', 'FontSize', 14)
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)')
saveas(gcf, 'Plots/monk3_computation_time.png')
