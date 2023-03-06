import cv2 as cv
import numpy as np

import keras

model = keras.models.load_model('mnist.h5')

draw = False

def draw_callback(event, x, y, flags, param):
    global draw
    if event == cv.EVENT_MOUSEMOVE:
        if draw:
            cv.circle(img, (x, y), 20, 200, -1)
    elif event == cv.EVENT_LBUTTONDOWN:
        draw = True
    elif event == cv.EVENT_LBUTTONUP:
        draw = False

img = np.zeros((512, 512), dtype="uint8")

cv.namedWindow("MNIST")
cv.setMouseCallback("MNIST", draw_callback)

while True:
    cv.imshow("MNIST", img)

    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord('m'):
        image = cv.resize(img, (28, 28))
        image = image / 255
        image = np.array([image])
        image = image.reshape(*image.shape, 1)

        prediction = model.predict(image)
        # print(prediction)
        num = np.argmax(prediction, 1)
        print(prediction[0, num], num[0])
    if key == ord('c'):
        img[:] = 0


cv.destroyAllWindows()
