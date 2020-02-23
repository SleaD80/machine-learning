function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
H1 = sigmoid([ones(m, 1) X] * Theta1');
H2 = sigmoid([ones(m, 1) H1] * Theta2');

%fprintf('#1\n')
  Y = zeros(m,num_labels);
  for i = 1:m
    Y(i,y(i)) = 1;
  end
  %Y = eye(num_labels)(y,:);
  J = sum(sum(-Y.*log(H2) - (1-Y).*log(1-H2)))/m;
  %fprintf('H1: %d*%d\n', size(H1,1), size(H1,2))
  %fprintf('H2: %d*%d\n', size(H2,1), size(H2,2))
  %fprintf('y: %d*%d\n', size(y,1), size(y,2))
  %fprintf('Y: %d*%d\n', size(Y,1), size(Y,2))
  %fprintf('J: %d*%d\n', size(J,1), size(J,2))
  J = J + lambda*(sum(sum([zeros(size(Theta1,1),1) Theta1(:,2:end)].^2))+sum(sum([zeros(size(Theta2,1),1) Theta2(:,2:end)].^2)))/(2*m);

% Grad calculation
  %fprintf('Theta1: %d*%d\n', size(Theta1,1), size(Theta1,2));
  %fprintf('Theta2: %d*%d\n', size(Theta2,1), size(Theta2,2));
  D_1 = zeros(size(Theta1));
  D_2 = zeros(size(Theta2));
  A1 = [ones(m, 1) X];
  %fprintf('A1: %d*%d\n', size(A1,1), size(A1,2))
  Z2 = A1 * Theta1';
  %fprintf('Z2: %d*%d\n', size(Z2,1), size(Z2,2))
  A2 = [ones(m, 1) sigmoid(Z2)];
  %fprintf('A2: %d*%d\n', size(A2,1), size(A2,2))
  Z3 = A2 * Theta2'; 
  %fprintf('Z3: %d*%d\n', size(Z3,1), size(Z3,2))
  A3 = sigmoid(Z3);
  %fprintf('A3: %d*%d\n', size(A3,1), size(A3,2))

  d3 = A3 - Y;
  %fprintf('d3: %d*%d\n', size(d3,1), size(d3,2))
  d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(Z2);
  %size(d2)
  %fprintf('d2: %d*%d\n', size(d2,1), size(d2,2))
  D1 = d2' * A1;
  %fprintf('D1: %d*%d\n', size(D1,1), size(D1,2))
  D2 = d3' * A2;
  %fprintf('D2: %d*%d\n', size(D2,1), size(D2,2))

  Theta1(:,1) = 0;
  Theta2(:,1) = 0;

  Theta1_grad = D1 / m + lambda * Theta1 / m;
  Theta2_grad = D2 / m + lambda * Theta2 / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
