clear ; close all; clc
hidden_layer_size = 25;
num_labels = 2; 

load('../data/feature_mat/feature_data_from_red.mat');
feature_mat = [vessel_feature_mat; non_vessel_feature_mat];
y = [ones(size(vessel_feature_mat,1),1)*2; ones(size(non_vessel_feature_mat,1),1)];
m = size(feature_mat, 1);

options = optimset('MaxIter', 100);
lambda = 1;
shuffle_index = randperm(m);
feature_mat = feature_mat(shuffle_index,:);  %Shuffle rows
y = y(shuffle_index,:);

scales = 3:5:50-1;
train_size = round(m*0.8);
trainy = y(1:train_size,:);
cv_y = y(train_size+1:end,:);
Jcv = zeros(1,3);

initial_Theta1 = randInitializeWeights(size(feature_mat,2), hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Third order
trainX = feature_mat(1:train_size,:);
cv_X = feature_mat(train_size+1:end,:);
costFunction = @(p) nnCostFunction(p, ...
                                   size(feature_mat,2), ...
                                   hidden_layer_size, ...
                                   num_labels, trainX, trainy, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

[J grad] = nnCostFuncNoReg(nn_params,size(feature_mat,2),hidden_layer_size,num_labels,cv_X,cv_y,lambda);
Jcv(3) = J;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Second order
initial_Theta1_order2 = zeros(hidden_layer_size,length(scales)*18+1);
initial_Theta1_order2(:,1) = initial_Theta1(:,1);
feature_mat_order2 = zeros(m,length(scales)*18);
for i = 1:length(scales)
     initial_Theta1_order2(:,(i-1)*18+1+1:(i-1)*18+18+1) = initial_Theta1(:,(i-1)*30+1+1:(i-1)*30+18+1);
     feature_mat_order2(:,(i-1)*18+1:(i-1)*18+18) = feature_mat(:,(i-1)*30+1:(i-1)*30+18);
end

initial_nn_params_order2 = [initial_Theta1_order2(:) ; initial_Theta2(:)];
trainX = feature_mat_order2(1:train_size,:);
cv_X = feature_mat_order2(train_size+1:end,:);
costFunction = @(p) nnCostFunction(p, ...
                                   size(feature_mat_order2,2), ...
                                   hidden_layer_size, ...
                                   num_labels, trainX, trainy, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params_order2, options);

[J grad] = nnCostFuncNoReg(nn_params,size(feature_mat_order2,2),hidden_layer_size,num_labels,cv_X,cv_y,lambda);
Jcv(2) = J;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% First order
initial_Theta1_order1 = zeros(hidden_layer_size,length(scales)*9+1);
initial_Theta1_order1(:,1) = initial_Theta1(:,1);
feature_mat_order1 = zeros(m,length(scales)*9);
for i = 1:length(scales)
     initial_Theta1_order1(:,(i-1)*9+1+1:(i-1)*9+9+1) = initial_Theta1(:,(i-1)*30+1+1:(i-1)*30+9+1);
     feature_mat_order1(:,(i-1)*9+1:(i-1)*9+9) = feature_mat(:,(i-1)*30+1:(i-1)*30+9);
end
initial_nn_params_order1 = [initial_Theta1_order1(:) ; initial_Theta2(:)];
trainX = feature_mat_order1(1:train_size,:);
cv_X = feature_mat_order1(train_size+1:end,:);
costFunction = @(p) nnCostFunction(p, ...
                                   size(feature_mat_order1,2), ...
                                   hidden_layer_size, ...
                                   num_labels, trainX, trainy, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params_order1, options);

[J grad] = nnCostFuncNoReg(nn_params,size(feature_mat_order1,2),hidden_layer_size,num_labels,cv_X,cv_y,lambda);
Jcv(1) = J;