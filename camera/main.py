import os
import numpy as np
import cv2

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

model = tf.keras.models.load_model('asl.h5')

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
               'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
               'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


def load_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
    img_tensor = tf.keras.utils.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return img_tensor


def predict():
    img_path = "test.jpg"
    new_image = load_image(img_path)
    result = model.predict(new_image)
    res = class_names[np.argmax(result)]
    return res


cap = cv2.VideoCapture(0)

res = ''
i = 0

brojac = 0
while True:
    ret, img = cap.read()

    if ret:
        x1, y1, x2, y2 = 200, 100, 400, 300
        img_cropped = img[y1:y2, x1:x2]

        cv2.imwrite('test.jpg', img_cropped)
        # cv2.imwrite('slike/test'+str(brojac)+'.jpg', img_cropped)

        a = cv2.waitKey(1)  # waits to see if `esc` is pressed

        if i == 4:
            res_tmp = predict()
            res = res_tmp
            i = 0

        i += 1
        brojac += 1

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, '%s' % (res.upper()), (100, 400), cv2.FONT_HERSHEY_DUPLEX, 4, (255, 255, 255), 4)
        cv2.imshow("Sign language recognition", img)

        if a == 27:  # when `esc` is pressed
            break

cv2.destroyAllWindows()
cv2.VideoCapture(0).release()
