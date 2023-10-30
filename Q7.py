from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import PartialDependenceDisplay

# One-layer MLP : you can use  learning_rate_init=0.001 to get a reasonable model, optimize other parameters by experimentation
# We advice that you name variable for the mlp regressor model 'mlp_reg' so that it will be consistent 
# with the scripts to call your implementation of PFI later in Q8:

learning_rate = ['constant', 'invscaling', 'adaptive']
mlp_reg = MLPRegressor(hidden_layer_sizes=(100), activation='relu', solver='adam', learning_rate=learning_rate[2], learning_rate_init=0.001, alpha = 0.0001)
mlp_reg.fit(train_features, train_labels)

