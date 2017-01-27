hidden_layer_size = 15;
num_labels = 2; 

fprintf('Loading Data ...\n');
load('../data/feature_mat/major_vessels_only-no_kamiyama/batch1.mat');
feature_mat = [vessel_feature_mat; non_vessel_feature_mat];
input_layer_size = size(vessel_feature_mat,2);
y = [ones(size(vessel_feature_mat,1),1)*2; ones(size(non_vessel_feature_mat,1),1)];
m = size(feature_mat, 1);

fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% unroll
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 100);
lambda = 1;
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, feature_mat, y, lambda);
    
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, feature_mat);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
save -6 ../data/nn_param/major_vessels_only-no_kamiyama.mat Theta1 Theta2