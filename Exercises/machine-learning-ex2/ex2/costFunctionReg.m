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


z = X*theta;
hx = sigmoid(z);
for i = 1:m
  J = J + (-y(i)*log(hx(i)) - (1 - y(i))*log(1 - hx(i)));
end
J = J/m;

% R = 0;
% n = size(X,2);
% for j = 1:n
  % R = R + lambda*(theta(j)^2)/(2*m);
% end
th = [0;theta(2:end)];
J = J + lambda*(th'*th)/(2*m);

grad = (X'*(sigmoid(X*theta) - y))/m + (lambda*th)/m;




% =============================================================

end
