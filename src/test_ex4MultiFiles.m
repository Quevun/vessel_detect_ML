clear ; close all; clc
hidden_layer_size = 25;
num_labels = 2; 

load('../data/feature_mat/more_non_vessels2/batch1.mat');
feature_mat = [vessel_feature_mat; non_vessel_feature_mat];
input_layer_size = size(vessel_feature_mat,2);
y = [ones(size(vessel_feature_mat,1),1)*2; ones(size(non_vessel_feature_mat,1),1)];
m = size(feature_mat, 1);

fprintf('\nInitializing Neural Network Parameters ...\n');
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n');
options = optimset('MaxIter', 100);
lambda = 1;

[J1 grad1] = nnCostFunction(initial_nn_params,input_layer_size,hidden_layer_size,...
                            num_labels,feature_mat,y,lambda);
                            
clear feature_mat y
[J2 grad2] = nnCostFuncMultiFiles(initial_nn_params,input_layer_size,...
                                  hidden_layer_size,num_labels,lambda);                                          

disp(J1==J2);
disp(isequal(grad1,grad2));