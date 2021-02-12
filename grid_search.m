function [best_beta, best_errors, alpha, lambda] = grid_search(Optimizer, Error, X, T, f, eps, N, W, b, beta)

    alpha_candidates = 0:0.01:0.1;
    lambda_candidates = 0:(0.01/100):0.001;
    
    best_error = Inf;
    best_errors = [];
    best_beta = 0;
    
    for al = 1:length(alpha_candidates)
        for ll = 1:length(lambda_candidates)
            a = alpha_candidates(al);
            l = lambda_candidates(ll);

            hessian = 0;
            for i = 1:N
                x = X(:,i);
                t = T(:,i);
                hidden_out = f(W * x + b);
                hessian = hessian + (hidden_out * hidden_out');
            end

            hessian = 2/N * (hessian + l);

            eta = 1/norm(hessian);

           [beta_nag, errs] = Optimizer(Error, beta, eps, eta, l, a, N, X, T, W, b, f, false, 2000);

           [e, ~] = Error(beta_nag, X, T, W, b, N, f, l);
           
           % if good then save hyperparameter conf.
           if e < best_error
              best_error = e;
              best_beta = beta_nag;
              best_errors = errs;
              alpha = a;
              lambda = l;
           end
        end
       fprintf('%d / %d hyperparameters tried. best error so far = %d\n', (al *length(lambda_candidates)), length(alpha_candidates) * length(lambda_candidates), best_error)
    end
    fprintf('Grid search result:\n')
    fprintf('alpha = %d\n', alpha)
    fprintf('lambda = %d\n', lambda)
end