import numpy as np
from sklearn.model_selection import train_test_split
import CreateDataset
import pickle
from AutoEncoder import encode


# Preparing train and test data
img_data, class_name = CreateDataset.create_dataset()
target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
target_val = [target_dict[class_name[i]] for i in range(len(class_name))]

X = img_data
y = target_val
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=7)

X_train = np.array(X_train).reshape(len(np.array(X_train)), np.prod(np.array(X_train).shape[1:]))
X_test = np.array(X_test).reshape(len(np.array(X_test)), np.prod(np.array(X_test).shape[1:]))

encoded_train_features = encode(X_train)
encoded_test_features = encode(X_test)

with open('./DataSet/train.pickle', 'wb') as f:
    pickle.dump([encoded_train_features, y_train], f)
with open('./DataSet/test.pickle', 'wb') as f:
    pickle.dump([encoded_test_features, y_test], f)
