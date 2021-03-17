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

nag_precision_times = (0);
bfgs_awls_precision_times = (0);
bfgs_bls_precision_times = (0);

iter = 1;
h_min = 101;
%h_max = 10000;
%step = 1000;
h_max = 101;
step = 1;

precision = 1e-6;

for h = h_min:step:h_max
    W = randn(h,n);         % weight between input and hidden layer
    b = randn(h,1);         % bias of hidden nodes
    beta = randn(h,m);      % randomly initialized beta
    
    
    hessian = 0;
    for i = 1:N
        x = X(:,i);
        t = T(:,i);
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
        %[beta_nag, errors_nag] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, false, intmax, 0);
        [beta_nag, errors_nag, ~, prec_tEnd] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, false, intmax, intmax, opt_val, precision, true);
    end
    tEnd = toc(ticStart);
    nag_precision_times(iter) = prec_tEnd;
    
    fprintf('\n-- NAG --\n')
    fprintf('#iter = %d\n', length(errors_nag))
    fprintf('final error = %d\n', errors_nag(length(errors_nag)))
    fprintf('time/iter = %d\n', (tEnd / k) / length(errors_nag))
    nag_times(iter) = (tEnd / k) / length(errors_nag);
    
    % --- Time BFGS (BLS)
    B = eye(h*m);
    ticStart = tic;
    for i = 1:k
        %[beta_bfgs_bls, errors_bfgs_bls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', false);
        [beta_bfgs_bls, errors_bfgs_bls, ~, ~, bls_prec_tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', false, opt_val, precision, true);
    end
    tEnd = toc(ticStart);
    bfgs_bls_precision_times(iter) = bls_prec_tEnd;
    %check if this is greater than 1/10s to get accuracy, otherwise
    %increase k
    %fprintf('BFGS time = %d\n', (tEnd))

    bfgs_bls_times(iter) = (tEnd / k) / length(errors_bfgs_bls);
    fprintf('\n-- BLS --\n')
    fprintf('#iter = %d\n', length(errors_bfgs_bls))
    fprintf('final error = %d\n', errors_bfgs_bls(length(errors_bfgs_bls)))
    fprintf('time/iter = %d\n', (tEnd / k) / length(errors_bfgs_bls))
    
    % --- Time BFGS (AWLS)
    B = eye(h*m);
    ticStart = tic;
    for i = 1:k
        %[beta_bfgs_awls, errors_bfgs_awls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS', false);
        [beta_bfgs_awls, errors_bfgs_awls, ~, ~, awls_prec_tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS', false, opt_val, precision, true);
    end
    tEnd = toc(ticStart);
    bfgs_awls_precision_times(iter) = awls_prec_tEnd;
    
    bfgs_awls_times(iter) = (tEnd / k) / length(errors_bfgs_awls);
    fprintf('\n-- AWLS --\n')
    fprintf('#iter = %d\n', length(errors_bfgs_awls))
    fprintf('final error = %d\n', errors_bfgs_awls(length(errors_bfgs_awls)))
    fprintf('time/iter = %d\n', (tEnd / k) / length(errors_bfgs_awls))
    
    fprintf('%d/%d\n', step*(iter-1), h_max-h_min)
    
    iter = iter + 1;
end

plot(h_min:step:h_max, nag_times, h_min:step:h_max, bfgs_bls_times, h_min:step:h_max, bfgs_awls_times)
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)')
%saveas(gcf, 'Plots/sin_timings.png')

figure

plot(h_min:step:h_max, nag_precision_times, h_min:step:h_max, bfgs_bls_precision_times, h_min:step:h_max, bfgs_awls_precision_times)
title('time [s] to get to precision vs m')
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)', 'Location', 'northwest')
saveas(gcf, 'Plots/sin_time_to_accuracy.png')
