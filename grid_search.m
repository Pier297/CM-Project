function [alpha, lambda] = grid_search(Optimizer, Error, X, T, f)

    alpha_candidates = 0:0.1:1;
    lambda_candidates = 0:(0.01/10):0.01;
    
    best_error = Inf;
    
    for al = 1:length(alpha_candidates)
       for ll = 1:length(lambda_candidates)
           a = alpha_candidates(al);
           l = lambda_candidates(ll);
           
           avg_errors = 0;
           k = 5;
           
           % For each hyperparameter conf, try multiple times to avoid
           % lucky random initialization
           for it = 1:k
                n = size(X,2);          % input dimension
                m = size(T,2);          % output dimension
                N = size(X,1);          % number of samples
                h = N;                  % number of hidden units  
                W = randn(h,n);         % weight between input and hidden layer
                b = randn(h,1);         % bias of hidden nodes
                beta = randn(h,m);      % randomly initialized beta
                eps = 1e-1;

                % Compute hessian
                hessian = 0;
                for i = 1:N
                    x = X(:,i);
                    t = T(:,i);
                    hidden_out = f(W * x + b);
                    hessian = hessian + (hidden_out * hidden_out');
                end

                hessian = 2/N * (hessian + l);

                eta = 1/norm(hessian);


                [beta_nag, ~] = Optimizer(Error, beta, eps, eta, l, a, N, X, T, W, b, f);

                [e, ~] = Error(beta_nag, X, T, W, b, N, f, l);
                
                avg_errors = avg_errors + e;
           end
           
           % if good then save hyperparameter conf.
           if avg_errors/k < best_error
              best_error = avg_errors/k;
              alpha = a;
              lambda = l;
           end
       end
    end
end