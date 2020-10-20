from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pickle


train_file = open('./DataSet/train.pickle', "rb")
X_train, y_train = pickle.load(train_file)

test_file = open('./DataSet/test.pickle', "rb")
X_test, y_test = pickle.load(test_file)

X_train = X_train.reshape(len(X_train), 78)
X_test = X_test.reshape(len(X_test), 78)
y_train = np.array(y_train)
y_test = np.array(y_test)

mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(100, 78),
                    random_state=5, verbose=True, learning_rate_init=.1, tol=1e-4, batch_size='auto')
mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

filename = './Models/Classifier/classifier_model.sav'
pickle.dump(mlp, open(filename, 'wb'))

y_pred = mlp.predict(X_test)

cmap = ListedColormap(['lightgrey', 'silver', 'ghostwhite', 'lavender', 'wheat'])


# confusion matrix
def cm(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    plt.matshow(cm, cmap=cmap)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x=j, y=i, s=cm[i, j], va='center', ha='center')

    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()


cm(y_train.argmax(1), mlp.predict(X_train).argmax(1), title='Train')
cm(y_test.argmax(1), mlp.predict(X_test).argmax(1), title='Test')
