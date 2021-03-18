The project implements two optimization algorithms: BFGS and NAG.

# Main Files

- ObjectiveFunc.m <br>
Implements the Mean-Squared-Error with L2 regularization. <br>
When called it computes the value and the gradient of the error at the specified point.

- true_solution.m <br>
Computes the true solution of the optimization problem by finding the root of the gradient of the error.

- NAG.m <br>
Optimizes the mean-squared-error using Nesterov's Accelerated Gradient.

- BFGS.m <br>
Optimizes the mean-squared-error using a quasi-Newton method. Supports two line searches: 'BLS' or 'AWLS'

___

Both NAG and BFGS can be executed with two differents stopping condition:
1. Stops when the relative norm of the gradient is less than or equal to 'eps'. <br>

        [beta_nag, ~, ~, ~] = NAG(@ObjectiveFunc, beta, eps, eta, lambda, N, X, T, W, b, f, true, intmax, intmax, -Inf, -1, false);


2. Keep going until it reaches a predefined relative error. Stops when the relative error is less than or equal to 'accuracy'. <br>

        [beta_nag, ~, ~, ~] = NAG(@ObjectiveFunc, beta, -1, eta, lambda, N, X, T, W, b, f, true, intmax, intmax, opt_val, accuracy, true);

# Tests

## sin(x) dataset
- test_sin.m <br>
A script to run the optimizations algorithms on the sin(x) dataset. It can also be used to control the noise in the data.

- sin_timings.m <br>
Measure the time taken per iteration and the total time to reach a predefined relative error for different number of hidden units h.

## MONK

- test_monk.m <br>
A script to run the optimizations algorithms on the various MONK problems.

- monk_timings.m <br>
Measure the time taken per iteration and the total time to reach a predefined relative error for different number of hidden units h.

## Random matrix

- test_random_matrix.m <br>
Tests on random matrix.

- random_timings.m <br>
Measure the time taken per iteration and the total time to reach a predefined relative error for different output dimensions m.

- ill_random_matrix.m <br>
Performs optimization tests on a close to singular random matrix.
