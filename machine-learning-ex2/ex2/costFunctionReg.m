function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
mask = ones(size(theta))'; % 1 x size(theta)
mask(1) = 0; % first term is 0, so we don't regularize theta_0
one = ones(m,1); % array made of ones (m x 1)
h_theta = sigmoid(X*theta); % (m x 1)
J = 1/m * ((- y' * log(h_theta)) - ((one - y)' * log((one - (h_theta))))) + ...
  lambda/(2*m) * sum((mask' .* theta) .* (mask' .* theta));
regularize_term = lambda/m .* mask; % 1 x size(theta)
grad = 1/m * (X' * (h_theta - y)) + regularize_term' .* theta;


% =============================================================

end
