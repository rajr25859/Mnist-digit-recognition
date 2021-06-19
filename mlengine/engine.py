import joblib
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import numpy as np
from mnist import MNIST
import os
import pickle
from sklearn.metrics import classification_report, accuracy_score


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

# print('predicted model')
# pred_knn = knn.predict(X_test)

# print('accuracy score')

# print(accuracy_score(y_test,pred_knn))

# print('classification report ')

# print(classification_report(y_test,pred_knn))

# print('load the new data')

# path = 'D:\DATA FOR MACHINE LEARNING PROJECT\MNIST_model_project\mlengine\pics prediction'
# l = os.listdir(path)
# img = []
# for i in l:
#     image = Image.open(f'D:\DATA FOR MACHINE LEARNING PROJECT\MNIST_model_project\mlengine\pics prediction\{i}')
#     # f_img = image.flatten().reshape(28, 28)
#     img.append(np.asarray(image).flatten())

# print('predicted testing data')
# model_pred_test = knn.predict(img)

# print('accuracy score of testing data')

# print(accuracy_score(y_test,model_pred_test))

# print('classification resport of testing data')

# print(classification_report(y_test, model_pred_test)) 




print('Dumping the Model')

file_name = 'webapp/model/knn.pickle'
# pickle.dump(knn, open(file_name, 'wb'))

joblib.dump(knn, os.path.join(os.getcwd(), 'webapp/model/knn.pickle'))

print('Model Dumped ')
