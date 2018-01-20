function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% caculate activiation values of 2nd layer using predefined weights
a2 = sigmoid(X * Theta1')

% Add ones to the a2 data matrix
a2 = [ones(size(a2,1), 1) a2];

% caculate activiation values of Output layer using predefined weights
h = sigmoid(a2 * Theta2')

% generate vector (max value, index of max value)
[p_max, i_max] = max(h, [], 2);
p = i_max;


% =========================================================================


end