rng(1);   

% --- parameter
f = @tanh;              % hidden activation function
n = 5;
eps = 1e-3;

N = 250;
h = N;                % number of hidden units
% --- end of parameter

X = randn(N, n);

W = rand(h,n)*2-1;      % weight between input and hidden layer, range in [-1,1]
b = rand(h,1)*2-1;      % bias of hidden nodes, range in [-1,1]
X = X';                 % transpose to make it easier

lambda = 0;


nag_times = (0);
bfgs_awls_times = (0);
bfgs_bls_times = (0);

iter = 1;
m_min = 1;
m_max = 10;
step = 1;

for m = m_min:step:m_max
    T = randn(N, m);
    T = T';
    beta = rand(h,m)*2-1;   % randomly initialized beta, range in [-1,1]
    
    hessian = 0;
    for i = 1:N
        x = X(:,i);
        hidden_out = f(W * x + b);
        hessian = hessian + (hidden_out * hidden_out');
    end
    hessian = 2/N * (hessian + lambda);
    eta = 1/norm(hessian);
    
    % Try k times and then take avg
    k = 1;
    
    % --- Time NAG
    ticStart = tic;
    for i = 1:k
        [beta_nag, errors_nag] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, false, intmax, 0);
    end
    tEnd = toc(ticStart);
    
    fprintf('\n-- NAG --\n')
    fprintf('#iter = %d\n', length(errors_nag))
    fprintf('final error = %d\n', errors_nag(length(errors_nag)))
    fprintf('time/iter = %d\n', (tEnd / k) / length(errors_nag))
    nag_times(iter) = (tEnd / k) / length(errors_nag);
    
    % --- Time BFGS (BLS)
    B = eye(h*m);
    ticStart = tic;
    for i = 1:k
        [beta_bfgs_bls, errors_bfgs_bls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', false);
    end
    tEnd = toc(ticStart);
    %check if this is greater than 1/10s to get accuracy, otherwise
    %increase k
    %fprintf('time = %d\n', (tEnd))
    bfgs_bls_times(iter) = (tEnd / k) / length(errors_bfgs_bls);
    fprintf('\n-- BLS --\n')
    fprintf('#iter = %d\n', length(errors_bfgs_bls))
    fprintf('final error = %d\n', errors_bfgs_bls(length(errors_bfgs_bls)))
    fprintf('time/iter = %d\n', (tEnd / k) / length(errors_bfgs_bls))
    
    % --- Time BFGS (AWLS)
    B = eye(h*m);
    ticStart = tic;
    for i = 1:k
        [beta_bfgs_awls, errors_bfgs_awls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS', false);
    end
    tEnd = toc(ticStart);
    
    bfgs_awls_times(iter) = (tEnd / k) / length(errors_bfgs_awls);
    fprintf('\n-- AWLS --\n')
    fprintf('#iter = %d\n', length(errors_bfgs_awls))
    fprintf('final error = %d\n', errors_bfgs_awls(length(errors_bfgs_awls)))
    fprintf('time/iter = %d\n', (tEnd / k) / length(errors_bfgs_awls))
    
    fprintf('%d/%d\n', step*(iter-1), m_max-m_min)
    
    iter = iter + 1;
end

plot(m_min:step:m_max, nag_times, m_min:step:m_max, bfgs_bls_times, m_min:step:m_max, bfgs_awls_times)
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)', 'Location', 'northwest')
saveas(gcf, 'Plots/random_timings.png')
