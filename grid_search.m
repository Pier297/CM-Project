function [best_beta, best_errors, lambda] = grid_search(Optimizer, Error, X, T, f, eps, N, W, b, beta)
    min_lambda = 0;
    max_lambda = 0.01;
    
    samples = 10;

    lambda_candidates = min_lambda:(max_lambda - min_lambda)/samples:max_lambda;
    
    best_error = Inf;
    best_errors = [];
    best_beta = 0;
    
    for ll = 1:length(lambda_candidates)
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

       [beta_nag, errs] = Optimizer(Error, beta, eps, eta, l, N, X, T, W, b, f, false, 2000, 5);

       [e, ~] = Error(beta_nag, X, T, W, b, N, f, l);
       
       fprintf('%d\t%d\n', l, e)
           
       % if good then save hyperparameter conf.
       if e < best_error
          best_error = e
          best_beta = beta_nag;
          best_errors = errs;
          lambda = l;
       end
    end
    
    fprintf('Grid search result:\n')
    fprintf('lambda = %d\n', lambda)
end