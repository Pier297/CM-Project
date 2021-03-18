function [beta, errors, tEnd, prec_tEnd] = NAG(E, beta, eps, eta, lambda, N, X, T, W, b, f, print_stat, MAX_ITER, MAX_UNLUCKY_STEPS, opt_val, precision, keep_going_until_precision)
% Nesterov' Accelerated Gradient Descent with L2 regularization
%
% Input:
%   E                 : objective function
%   beta              : initial weight, a matrix with size = (h, m)
%   B                 : initial approximation of inverse Hessian, size = (h*m, h*m)
%   eps               : accuracy for stopping criterion
%   eta               : Fixed step size
%   lambda            : L2 regularization parameter
%   N                 : Number of training samples
%   X                 : Training data inputs
%   T                 : Training data outputs
%   W, b, f           : Defines the ELM
%   print_stat        : boolean flag, when equal to true it print info
%   MAX_ITER          : Stops when the number of iterations exceed this
%                       constant.
%   MAX_UNLUCKY_STEPS : Stops when the absolute error keep increasing after
%                       'MAX_UNLUCKY_STEPS' iterations. When this is equal to 0 it forces a
%                        minimizing sequence.
%   opt_val           : Optimal value of the problem, only used when
%                      'keep_going_until_precision' equals to true
%   precision         : Sets prefered relative error, keep the iterations going
%                       until the current relative error is greater than this 'precision'; only used when
%                       'keep_going_until_precision' equals to true
%   keep_going_until_precision : When it is equal to false it uses the
%                                relative norm of the gradient as a stopping condition, otherwise it
%                                uses the 'precision' condition.
%
% Outputs:
%   beta      : Optimal beta found by the optimization algorithm
%   errors    : List of the absolute errors computed at each iteration
%   tEnd      : Total number of seconds taken to found the optimal 'beta'.
%   prec_tEnd : Total number of seconds taken to reach 'precision'.

    function [a_t_plus_1] = update_a(a_t)
       a_t_plus_1 = (1 + sqrt(4 * a_t^2 + 1))/2;
    end

a_t_minus_1 = 1;
delta_beta_t_minus_1 = 0;

[v,gr] = E(beta, X, T, W, b, N, f, lambda);
ng0 = norm(gr);

errors = (v);
iter = 0;

prevError = v;
unluckySteps = 0;

tStart = tic;
prec_tStart = tic;
if keep_going_until_precision == false
    prec_tEnd = toc(prec_tStart);
end
got_precision = false;
while unluckySteps <= MAX_UNLUCKY_STEPS && iter < MAX_ITER && ((keep_going_until_precision == false && norm(gr) > eps * ng0) || (keep_going_until_precision == true && got_precision == false))
    a_t = update_a(a_t_minus_1);
    
    alpha_t_minus_1 = (a_t_minus_1 - 1)/a_t;

    [~, g] = E(beta + alpha_t_minus_1 * delta_beta_t_minus_1, X, T, W, b, N, f, lambda);

    delta_beta_t = alpha_t_minus_1 * delta_beta_t_minus_1 - eta * g;

    beta = beta + delta_beta_t;
    
    a_t_minus_1 = a_t;
    delta_beta_t_minus_1 = delta_beta_t;
    
    [v,gr] = E(beta, X, T, W, b, N, f, lambda);
    
    if (v - opt_val) <= precision
       prec_tEnd = toc(prec_tStart);
       got_precision = true;
    end
    
    if v >= prevError
        unluckySteps = unluckySteps + 1;
    else
        unluckySteps = 0;
    end
    
    prevError = v;

    iter = iter + 1;
    errors(iter+1) = v;
end
tEnd = toc(tStart);

if print_stat
    fprintf('\n### NAG ###\n')
    fprintf('# iterations = %d\nFinal error = %d\nElapsed time/iteration = %d\n', iter, v, tEnd / iter);
    
    % Test all decreasing errors
    all_decreasing = true;
    for i = 1:iter-1
        if errors(i) < errors(i+1)
            all_decreasing = false;
            break;
        end
    end
    if all_decreasing
        fprintf('The errors were all decreasing.\n')
    else
        fprintf('The errors were *NOT* all decreasing.\n')
    end
end

end
