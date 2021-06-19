import joblib
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from mnist import MNIST
import os

# import logging
# logging.basicConfig(filename = 'mlengine.log')
# Log = logging.getLogger('mlengine.engine')


print('Loading MNIST Data')
mnist_data = MNIST(os.path.join(os.getcwd(), 'mlengine/mnist_data'))
X_train, y_train = mnist_data.load_training()
X_test , y_test = mnist_data.load_testing()


X_train = np.asarray(X_train)

y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


print('Creating the Model')
knn = KNeighborsClassifier()

print('Training the Model')
knn.fit(X_train, y_train)

# print('Getting the Score')
# print('Training Score: {0}'.format(knn.score(X_train, y_train)))
# print('testing Score: {0}'.format(knn.score(X_test,y_test)))


print('Dumping the Model')
joblib.dump(knn, os.path.join(os.getcwd(), 'webapp/model/knn.pickle'))
print('Model Dumped ')
 