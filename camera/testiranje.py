import os

import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

img_height=64
img_width=64
#test_path = "testdataset/"
test_path = "testdataset2/"
num_of_classes = 29


class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
               'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
               'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

path = test_path
dir_list = os.listdir(path)

print(dir_list)


model = tf.keras.models.load_model('asl.h5')
actual=[]
pred=[]

for i in dir_list:
    actual.append(i.split('_')[1].split('.')[0])
    test_image = load_img(path+i, target_size = (img_width, img_height))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print(i+" predicted: "+class_names[np.argmax(result)]+", actual: "+i.split('_')[0])

    pred.append(class_names[np.argmax(result)])


print("Test accuracy=",accuracy_score(actual,pred))
print("Classification report:\n",classification_report(actual,pred))
cm = confusion_matrix(actual, pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm.diagonal())