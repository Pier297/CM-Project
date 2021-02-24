rng(1);

f   = @tanh;
eps = 1e-7;

% ---------------- sin(x) experiments ------------------ %

[X, T_Noise, T_Real] = create_sin_dataset(0, 10, 100, 0.5);

n = size(X,2);
m = size(T_Noise,2);
N = size(X,1);
h = N;

% transposes to make them easier
X = X';
T_Noise = T_Noise';
T_Real = T_Real';

[W, b, beta] = create_elm(n, h, m);

%[~, ~, lambda] = grid_search(@NAG, @ObjectiveFunc, X, T_Noise, f, eps, N, W, b, beta, X, T_Real);

%fprintf('best lambda = %d\n', lambda)

lambda = 1;

%plot_sin(X, T_Noise, T_Real, [], [], [], [], f, W, b, N, 'Plots/sin_noise_data.png')

[beta_opt, opt_val, opt_val_grad] = true_solution(X, T_Noise, W, b, f, N, h, m, lambda);
fprintf('true solution MSE = %d\n', opt_val)

%plot_sin(X, T_Noise, T_Real, beta_opt, [], [], [], f, W, b, N, 'Plots/sin_noise_nolambda_effect.png')


%eta = compute_eta(f, W, b, X, T_Noise, N, lambda);
%[beta_nag, errors_nag] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T_Noise, W, b, f, false, intmax, intmax);

%[v, ~] = ObjectiveFunc(beta_nag, X, T_Real, W, b, N, f, lambda);

%fprintf('NAG\n')
%fprintf('MSE noise = %d\n', errors_nag(length(errors_nag)))
%fprintf('MSE real = %d\n', v)
%fprintf('#iter = %d\n', length(errors_nag))

%plot_sin(X, T_Noise, T_Real, [], beta_nag, [], [], f, W, b, N, '')

B = eye(h*m);
[beta_bfgs_bls, errors_bfgs_bls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T_Noise, lambda, N, 'BLS', false);

[v, ~] = ObjectiveFunc(beta_bfgs_bls, X, T_Real, W, b, N, f, lambda);

fprintf('\nBFGS (BLS) \n')
fprintf('MSE noise = %d\n', errors_bfgs_bls(length(errors_bfgs_bls)))
fprintf('MSE real = %d\n', v)
fprintf('#iter = %d\n', length(errors_bfgs_bls))


%B = eye(h*m);
%[beta_bfgs_awls, errors_bfgs_awls] = BFGS(@ObjectiveFunc, beta, B, eps, h, m, W, b, f, X, T_Noise, lambda, N, 'AWLS', false);

%[v, ~] = ObjectiveFunc(beta_bfgs_awls, X, T_Real, W, b, N, f, lambda);

%fprintf('\nBFGS (AWLS) \n')
%fprintf('MSE noise = %d\n', errors_bfgs_awls(length(errors_bfgs_awls)))
%fprintf('MSE real = %d\n', v)
%fprintf('#iter = %d\n', length(errors_bfgs_awls))


plot_sin(X, T_Noise, T_Real, [], [], beta_bfgs_bls, [], f, W, b, N, '')

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
    W    = randn(h, n);
    b    = randn(h, 1);
    beta = randn(h, m);
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
        T_Noise(c) = sin(i) + correction;
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
        legends = [legends, "BFGS (BLS)"];
        plot(X, Y)
    end
    
    if isempty(beta_bfgs_awls) == false
        hold on
        Y = predict_sin(X, beta_bfgs_awls, f, W, b, N);
        legends = [legends, "BFGS (AWLS)"];
        plot(X, Y)
    end
    
    legend(legends, 'Location', 'southwest')
    
    if isempty(saveto) == false
       saveas(gcf, saveto) 
    end
end