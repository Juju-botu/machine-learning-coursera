function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
  delta1 = 0; 
  delta2 = 0;
  for k = 1:m
    delta1 += (theta' * X(k, :)' - y(k)) * X(k,1);
    delta2 += (theta' * X(k, :)' - y(k)) * X(k,2); 
  endfor
  theta(1,1) = theta(1,1) -(alpha/m) * delta1;
  theta(2,1) = theta(2,1) -(alpha/m) * delta2;
% NOT WORKING: differs of not much. Why? Bad rounding?
%   for i = 1:m
%      for j = 1:2
%       theta(j, 1) = theta(j, 1) - alpha/m * (theta' * X(i, :)' - y(i, 1)) * X(i, j); % vectorized implementation
%      endfor   
%    endfor


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
