rng(1);                 % seed to make random values repeatable

X = zeros(100, 1);
T = zeros(100, 1);
c = 1;
for i = 0:0.1:10
    X(c) = i;
    T(c) = sin(i); %+ (rand - 0.5)/3;
    c = c + 1;
end

f = @tanh;                    % hidden activation function
eps = 1e-6;

n = size(X,2);          % input dimension
m = size(T,2);          % output dimension
N = size(X,1);          % number of samples
X = X';                 % transpose to make it easier
T = T';                 % transpose to make it easier
lambda = 0;

nag_times = (0);
bfgs_awls_times = (0);
bfgs_bls_times = (0);

iter = 1;
h_min = 100;
h_max = 200;
step = 10;

for h = h_min:step:h_max
    beta = randn(h,m);      % randomly initialized beta
    W = randn(h,n);         % weight between input and hidden layer
    b = randn(h,1);         % bias of hidden nodes
    
    hessian = 0;
    for i = 1:N
        x = X(:,i);
        t = T(:,i);
        hidden_out = f(W * x + b);
        hessian = hessian + (hidden_out * hidden_out');
    end
    hessian = 2/N * (hessian + lambda);
    eta = 1/norm(hessian);
    
    % Try k times and then take avg
    k = 30;
    
    % --- Time NAG
    ticStart = tic;
    for i = 1:k
        [beta_nag, errors_nag] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, false, intmax, 0);
    end
    tEnd = toc(ticStart);
    
    nag_times(iter) = tEnd / k;
    
    % --- Time BFGS (BLS)
    B = eye(h*m);
    ticStart = tic;
    for i = 1:k
        [beta_bfgs_bls, errors_bfgs_bls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', false);
    end
    tEnd = toc(ticStart);
    
    bfgs_bls_times(iter) = tEnd / k;
    
    % --- Time BFGS (AWLS)
    B = eye(h*m);
    ticStart = tic;
    for i = 1:k
        [beta_bfgs_awls, errors_bfgs_awls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS', false);
    end
    tEnd = toc(ticStart);
    
    bfgs_awls_times(iter) = tEnd / k;
    
    fprintf('%d/%d\n', step*(iter-1), h_max-h_min)
    
    iter = iter + 1;
end

plot(h_min:step:h_max, nag_times, h_min:step:h_max, bfgs_bls_times, h_min:step:h_max, bfgs_awls_times)
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)')
saveas(gcf, 'Plots/sin_timings.png')
