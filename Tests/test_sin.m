rng(1);

f   = @tanh;
eps = 1e-6;


[X, T_Noise, T_Real] = create_sin_dataset(0, 10, 100, 0);

n = size(X,2);
m = size(T_Noise,2);
N = size(X,1);
h = N;

% transposes to make them easier
X = X';
T_Noise = T_Noise';
T_Real = T_Real';

[W, b, beta] = create_elm(n, h, m);

lambda = 0;

fprintf('lambda = %d\n', lambda)

fprintf('true solution\n')

%plot_sin(X, T_Noise, T_Real, [], [], [], [], f, W, b, N, 'Plots/sin_noise_data.png')

[beta_opt, opt_val_noise, opt_val_grad] = true_solution(X, T_Noise, W, b, f, N, h, m, lambda);
fprintf('NOISE MSE = %d\n', opt_val_noise)

[opt_val_real, ~] = ObjectiveFunc(beta_opt, X, T_Real, W, b, N, f, lambda);
fprintf('REAL MSE = %d\n', opt_val_real)

%plot_sin(X, T_Noise, T_Real, beta_opt, [], [], [], f, W, b, N, 'Plots/sin_noise_nolambda_effect.png')

% ---------------- sin(x) experiments ------------------ %


precision = 1e-1; % relative error upper bound


%[~, ~, lambda] = grid_search(@NAG, @ObjectiveFunc, X, T_Noise, f, eps, N, W, b, beta, X, T_Real);

%fprintf('best lambda = %d\n', lambda)

eta = compute_eta(f, W, b, X, T_Noise, N, lambda);
[beta_nag, errors_nag, ~, ~] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T_Noise, W, b, f, true, intmax, intmax, opt_val_noise, precision, true);
%[beta_nag, errors_nag, ~, ~] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T_Noise, W, b, f, true, intmax, intmax, opt_val, precision, true);


B = eye(h*m);
[beta_bfgs_bls, errors_bfgs_bls, ~, ~, bls_prec_tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T_Noise, lambda, N, 'BLS', true, opt_val_noise, precision, true);
%[beta_bfgs_bls, errors_bfgs_bls, ~, ~, bls_prec_tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T, lambda, N, 'BLS', true, opt_val, precision, true);
    

B = eye(h*m);
%[beta_bfgs_awls, errors_bfgs_awls, ~, ~, ~] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T_Noise, lambda, N, 'AWLS', true, opt_val_noise, precision, true);
[beta_bfgs_awls, errors_bfgs_awls, ~, ~, awls_prec_tEnd] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T_Noise, lambda, N, 'AWLS', true, opt_val_noise, precision, true);
    

%[v, ~] = ObjectiveFunc(beta_bfgs_awls, X, T_Real, W, b, N, f, lambda);
%fprintf('REAL MSE = %d %d\n', v, v - opt_val_real)

%plot_sin(X, T_Noise, T_Real, [], [], no_lambda_beta_bfgs_awls, lambda_beta_bfgs_awls, f, W, b, N, 'Plots/sin_noise_1e-5_lambda_bfgs.png')

%figure
%semilogy(1:(length(errors_nag)), errors_nag, 1:(length(errors_bfgs_awls)), errors_bfgs_awls, 1:(length(errors_bfgs_bls)), errors_bfgs_bls)
%xlabel('iteration', 'FontSize', 14)
%ylabel('log(Error)', 'FontSize', 14)
%legend('NAG', 'BFGS (BLS)', 'BFGS (AWLS)')
%saveas(gcf, 'Plots/10-6_100_sin_convergence_rate.png')

%figure
%semilogy(1:(length(errors_bfgs_awls)), errors_bfgs_awls)
%xlabel('iteration', 'FontSize', 14)
%ylabel('log(Error)', 'FontSize', 14)
%legend('BFGS (AWLS)')
%saveas(gcf, 'Plots/10-6_100_sin_BFGS_only_convergence_rate.png')

% ------------------------------------------------------ % 

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
