function beta = normal_equation(X, T, W, b, N, h, f)

tStart = tic;

H = zeros(N, h);
for j = 1:N
    for i = 1:h
        H(j, i) = f(transpose(W(i)) * transpose(X(j)) + b(i));
    end
end

beta = pinv(H) * T;

tEnd = toc(tStart);

fprintf('\n### Normal Equation ###\n')
fprintf('Time elapsed = %d\n', tEnd);

end

