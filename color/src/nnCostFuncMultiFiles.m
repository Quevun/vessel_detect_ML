function [J_sum grad] = nnCostFuncMultiFiles(nn_params, ...
                                         input_layer_size, ...
                                         hidden_layer_size, ...
                                         num_labels, ...
                                         lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
J_sum = 0;

files = dir('../data/feature_mat/more_non_vessels2/*.mat');
for file = files'
  load(strcat('../data/feature_mat/more_non_vessels2/',file.name));
  X = [vessel_feature_mat; non_vessel_feature_mat];
  y = [ones(size(vessel_feature_mat,1),1)*2; ones(size(non_vessel_feature_mat,1),1)];
  m = size(X, 1);
  
  X_bias = [ones(m,1) X];
  z2 = Theta1*X_bias';
  a2 = sigmoid(z2); %second activation units
  a2_bias = [ones(1,m); a2];
  hypo = sigmoid(Theta2*a2_bias);
  
  yvect = repmat(1:num_labels,m,1);
  yvect(:,1) = yvect(:,1) == y;
  yvect(:,2) = yvect(:,2) == y;

  JMatrix = -yvect'.*log(hypo) - (1-yvect').*log(1-hypo);
  JregTerm = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * lambda / (2*m);
  J = sum(sum(JMatrix)) / m + JregTerm;
  J_sum += J;

  error3 = hypo - yvect';
  error2 = (Theta2(:,2:end)'*error3).*sigmoidGradient(z2);

  delta1 = delta1 + error2 * X_bias;
  delta2 = delta2 + error3 * a2_bias';
end

Theta1_regTerm = Theta1;
Theta1_regTerm(:,1) = zeros(hidden_layer_size,1);
Theta2_regTerm = Theta2;
Theta2_regTerm(:,1) = zeros(num_labels,1);
Theta1_grad = delta1/m + Theta1_regTerm*lambda/m;
Theta2_grad = delta2/m + Theta2_regTerm*lambda/m;

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
