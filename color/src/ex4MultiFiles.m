clear ; close all; clc
input_layer_size = 180;
hidden_layer_size = 25;
num_labels = 2; 

fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 100);
lambda = 1;
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFuncMultiFiles(p, ...
                                         input_layer_size, ...
                                         hidden_layer_size, ...
                                         num_labels,lambda);
                                   
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

save -6 ../data/nn_param/more_non_vessels2.mat Theta1 Theta2