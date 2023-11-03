# Global explanations
import graphviz
from interpret import show

# Local explanations (LIME)
from interpret.blackbox import LimeTabular

import random

random.seed(6)
rand_index1 = random.randint(0, len(test_features))
rand_index2 = random.randint(0, len(test_features))

print(test_labels[rand_index1 : rand_index1+1])
print(mlp_reg.predict(test_features[rand_index1 : rand_index1+1]))

test_features_1 = test_features[rand_index1:rand_index1+1]
test_features_2 = test_features[rand_index2:rand_index2+1]
two_test_features = pd.concat([test_features_1, test_features_2])
print(two_test_features)

test_labels_1 = test_labels[rand_index1:rand_index1+1]
test_labels_2 = test_labels[rand_index2:rand_index2+1]
two_test_labels = pd.concat([test_labels_1, test_labels_2])
print(two_test_labels)

#Blackbox explainers need a predict function, and optionally a dataset
lime = LimeTabular(mlp_reg, test_features, random_state=1)

#Pick the instances to explain, optionally pass in labels if you have them
lime_local = lime.explain_local(two_test_features, two_test_labels, name='LIME')

show(lime_local, 0)
