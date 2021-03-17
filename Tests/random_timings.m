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

nag_precision_times = (0);
bfgs_awls_precision_times = (0);
bfgs_bls_precision_times = (0);

iter = 1;
m_min = 1;
m_max = 10;
step = 1;

precision = 1e-5;

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
    
    [~, opt_val, ~] = true_solution(X, T, W, b, f, N, h, m, lambda);
    fprintf('opt val = %d\n', opt_val)
    
    % Try k times and then take avg
    k = 1;
    
    % --- Time NAG
    ticStart = tic;
    for i = 1:k
        [beta_nag, errors_nag, ~, prec_tEnd] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, false, 500000, intmax, opt_val, precision, true);
    end
    tEnd = toc(ticStart);

    nag_times(iter) = (tEnd / k) / length(errors_nag);
    nag_precision_times(iter) = prec_tEnd;
    
    % --- Time BFGS (BLS)
    B = eye(h*m);
    ticStart = tic;
    for i = 1:k
        [beta_bfgs_bls, errors_bfgs_bls, ~, bls_prec_tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', false, opt_val, precision, true);
    end
    tEnd = toc(ticStart);

    bfgs_bls_times(iter) = (tEnd / k) / length(errors_bfgs_bls);
    bfgs_bls_precision_times(iter) = bls_prec_tEnd;
    
    % --- Time BFGS (AWLS)
    B = eye(h*m);
    ticStart = tic;
    for i = 1:k
        [beta_bfgs_awls, errors_bfgs_awls, ~, awls_prec_tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS', false, opt_val, precision, true);
    end
    tEnd = toc(ticStart);
    
    bfgs_awls_times(iter) = (tEnd / k) / length(errors_bfgs_awls);
    bfgs_awls_precision_times(iter) = awls_prec_tEnd;
    
    fprintf('%d/%d\n', step*(iter-1), m_max-m_min)
    
    iter = iter + 1;
end


plot(m_min:step:m_max, nag_times, m_min:step:m_max, bfgs_bls_times, m_min:step:m_max, bfgs_awls_times)
title('time [s] per iteration vs m')
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)', 'Location', 'northwest')
saveas(gcf, 'Plots/random_timings_2.png')

figure

plot(m_min:step:m_max, nag_precision_times, m_min:step:m_max, bfgs_bls_precision_times, m_min:step:m_max, bfgs_awls_precision_times)
title('time [s] to get to precision vs m')
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)', 'Location', 'northwest')
saveas(gcf, 'Plots/random_scalability_time_till_accuracy_2.png')
