function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%set possible variable values: 

possibleCs = [0 0.01 0.03 0.1 0.3 0.1 3 10 30 100]';
possibleSigmas = [0 0.01 0.03 0.1 0.3 0.1 3 10 30 100]';

%create a 10 by 10 matrix for holding the error

errorMatrix = zeros(10,10);

%loop through all the possible combinations to get the errormatrix of each combination of C and sigma:

for i = 1:length(possibleCs)
    for j =1:length(possibleSigmas)
        model = svmTrain(X,y, possibleCs(i), @(x1, x2) gaussianKernel(x1, x2, possibleSigmas(j)))
        error = mean(double(svmPredict(model, Xval) ~= yval))
        errorMatrix(i,j) = error;        
    end
end

%find the minimum value of the error matrix by turning the matrix into a vector

[M,I] = min(errorMatrix(:))

%find find index values of minimum to define C and sigma

[I_row, I_col] = ind2sub(size(errorMatrix),I)

C = possibleCs(I_row)
sigma = possibleSigmas(I_col)


% =========================================================================

end
