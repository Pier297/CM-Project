
input = load('monk1-train.txt');

[len, cols] = size(input);
x = input(1:len, 1:cols-1);
t = input(1:len, cols:cols);

% n is input dimension
n = size(x); n = n(1,2);
% m is output dimension
m = size(t); m = m(1,2);
% N is number of samples
N = size(x); N = N(1,1);
T = t;

% h is number of hidden units
h = N;

% Step 1
w = randn(h, n);
b = randn(h, 1);

% Step 2
H = zeros(N, h);
for j = 1:N
    for i = 1:h
        H(j, i) = tanh(transpose(w(i)) * transpose(x(j)) + b(i));
    end
end

% Step 3
B = pinv(H) * T;

model_out = H*B;

% Stats
correct = 0;
for i = 1:N
   if model_out(i) >= 0
       if t(i) == 0.9
           correct = correct + 1;
       end
   else
       if t(i) == -0.9
           correct = correct + 1;
       end
   end
end

correct

missclassifcations = N - correct

norm(H*B-t)

