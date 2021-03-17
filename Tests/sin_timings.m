% --- parameter
f = @tanh;
eps = 1e-6;
lambda = 0;
precision = 1e-7;
k = 3;
h_min = 0;
h_max = 1000;
step = 100;
% --- end of parameter

rng(1);
[X, ~, T] = create_sin_dataset(0, 10, 100, 0);
n = size(X,2);
m = size(T,2);
N = size(X,1);
X = X';
T = T';

nag_times = (0);
bfgs_awls_times = (0);
bfgs_bls_times = (0);

iter = 1;

for h = h_min:step:h_max
    [W, b, beta] = create_elm(n, h, m);
    
    % ------- True Solution -------
    [beta_opt, opt_val, opt_val_grad] = true_solution(X, T, W, b, f, N, h, m, lambda);
    fprintf('opt val = %d\n', opt_val)

    % --- NAG
    eta = compute_eta(f, W, b, X, T, N, lambda);
    for i = 1:k
        [~, errors_nag, ~, tEnd] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, false, intmax, intmax, opt_val, precision, true);
    end
    nag_times(iter) = (tEnd / k); % / length(errors_nag);

    % --- BFGS (BLS)
    for i = 1:k
        B = eye(h*m);
        [~, errors_bfgs_bls, ~, tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', false, opt_val, precision, true);
    end
    bfgs_bls_times(iter) = (tEnd / k); % / length(errors_bfgs_bls);

    % --- BFGS (AWLS)
    for i = 1:k
        B = eye(h*m);
        [~, errors_bfgs_awls, ~, tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'AWLS', false, opt_val, precision, true); 
    end
    bfgs_awls_times(iter) = (tEnd / k); % / length(errors_bfgs_awls);
    
    fprintf('%d/%d\n', step*(iter-1), h_max-h_min)
    iter = iter + 1;
end

plot(h_min:step:h_max, nag_times, h_min:step:h_max, bfgs_bls_times, h_min:step:h_max, bfgs_awls_times)
xlabel('Number of hidden nodes', 'FontSize', 14)
ylabel('Computation time', 'FontSize', 14)
legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)')
saveas(gcf, 'Plots/sin_scal_to_accuracy.png')




function [elm_out] = out(f, W, b, beta, x)
    elm_out = beta' * f(W * x + b);
end

function [eta] = compute_eta(f, W, b, X, T, N, lambda)
    hessian = 0;
    for i = 1:N
        x = X(:,i);
        t = T(:,i);
        hidden_out = f(W * x + b);
        hessian = hessian + (hidden_out * hidden_out');
    end
    hessian = 2/N * (hessian + lambda);
    eta = 1/norm(hessian);
end

% n = input dimension
% h = number of hidden units
% m = output dimension
function [W, b, beta] = create_elm(n, h, m)
    W    = randn(h, n)*2-1;
    b    = randn(h, 1)*2-1;
    beta = randn(h, m)*2-1;
end

% define N_samples points of the form (x, sin(x) +- noise)
function [X, T_Noise, T_Real] = create_sin_dataset(min_x, max_x, N_samples, noise)
    X       = zeros(N_samples, 1);
    T_Noise = zeros(N_samples, 1);
    T_Real  = zeros(N_samples, 1);
    c       = 1;
    
    for i = min_x:(max_x - min_x)/N_samples:max_x
        X(c) = i;
        % noise = 0.5 => sin(x) \in (-1.5, 1.5)
        min_noise  = -noise;
        max_noise  = noise;
        correction = min_noise + rand * (max_noise - min_noise);
        T_Noise(c) = (sin(i) + correction);
        T_Real(c)  = sin(i);
        c = c + 1;
    end
end

function [Y] = predict_sin(X, beta, f, W, b, N)
    Y = [];
    for i = 1:N
       Y = [Y, out(f, W, b, beta, X(:, i))]; 
    end
end

function plot_sin(X, T_Noise, T_Real, beta_opt, beta_nag, beta_bfgs_bls, beta_bfgs_awls, f, W, b, N, saveto)
    figure
    legends = ["sin(x)"];
    scatter(X, T_Noise)
    xlabel('x', 'FontSize', 14)
    ylabel('sin(x)', 'FontSize', 14)
    
    if isempty(T_Real) == false
        hold on
        scatter(X, T_Real)
        legends = [legends, "sin(x) without noise"];
    end
    
    
    if isempty(beta_opt) == false
        hold on
        Y = predict_sin(X, beta_opt, f, W, b, N);
        legends = [legends, "True Solution"];
        plot(X, Y)
    end
    
    if isempty(beta_nag) == false
        hold on
        Y = predict_sin(X, beta_nag, f, W, b, N);
        legends = [legends, "NAG"];
        plot(X, Y)
    end
    
    if isempty(beta_bfgs_bls) == false
        hold on
        Y = predict_sin(X, beta_bfgs_bls, f, W, b, N);
        legends = [legends, "BFGS (AWLS) lambda = 0"];
        plot(X, Y)
    end
    
    if isempty(beta_bfgs_awls) == false
        hold on
        Y = predict_sin(X, beta_bfgs_awls, f, W, b, N);
        legends = [legends, "BFGS (AWLS) lambda = 1e-5"];
        plot(X, Y)
    end
    
    legend(legends, 'Location', 'southwest')
    
    if isempty(saveto) == false
       saveas(gcf, saveto) 
    end
end