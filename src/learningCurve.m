clear ; close all; clc
hidden_layer_size = 50;
num_labels = 2; 

fprintf('Loading Data ...\n');
load('../data/feature_mat/feature2.mat');
feature_mat = [vessel_feature_mat; non_vessel_feature_mat];
input_layer_size = size(vessel_feature_mat,2);
y = [ones(size(vessel_feature_mat,1),1)*2; ones(size(non_vessel_feature_mat,1),1)];
m = size(feature_mat, 1);

fprintf('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% unroll
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n');
options = optimset('MaxIter', 100);
lambda = 1;
shuffle_index = randperm(m);
feature_mat = feature_mat(shuffle_index,:);  %Shuffle rows
y = y(shuffle_index,:);

train_size_max = round(m*0.8);
trainX = feature_mat(1:train_size_max,:);
trainy = y(1:train_size_max,:);
cv_X = feature_mat(train_size_max+1:end,:);
cv_y = y(train_size_max+1:end,:);
Jtrain = zeros(1,length(400:200:m));
Jcv = zeros(1,length(400:200:m));
i = 1;

for train_size = 400:200:train_size_max
     % Create "short hand" for the cost function to be minimized
     costFunction = @(p) nnCostFunction(p, ...
                                        input_layer_size, ...
                                        hidden_layer_size, ...
                                        num_labels, trainX(1:train_size,:), trainy(1:train_size,:), lambda);
                                        
     % Now, costFunction is a function that takes in only one argument (the
     % neural network parameters)
     [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
     
     [J grad] = nnCostFuncNoReg(nn_params,input_layer_size,hidden_layer_size,num_labels,trainX(1:train_size,:),trainy(1:train_size,:),lambda);
     Jtrain(i) = J;
     [J grad] = nnCostFuncNoReg(nn_params,input_layer_size,hidden_layer_size,num_labels,cv_X,cv_y,lambda);
     Jcv(i) = J;
     i += 1;
end
plot(400:200:train_size_max,Jtrain,400:200:train_size_max,Jcv);