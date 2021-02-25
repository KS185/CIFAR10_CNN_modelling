'''
predict a photo from CNN_CIFAR10 model
'''
import h5py
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt


category=['airplane','automobile', 'bird','cat','deer','dog','frog','horse',
          'ship','truck']
code=['0','1','2','3','4','5','6','7','8','9']
labels = dict(zip(code, category))

img_unsplash=cv2.imread('picture_bird.jpg')
img_unsplash=img_unsplash[50:400,100:550]
plt.imshow(img_unsplash)

image = cv2.resize(img_unsplash, dsize=(32,32),interpolation=cv2.INTER_CUBIC)
image=image.reshape(-1,32,32,3)

model = load_model('modeldata_new.hdf5')
 
pred=model.predict(image)
max1=pred.max()
pred_pos=np.where(pred[0]==max1)
pos_label=pred_pos[0][0]
print('--------------------------------------------')
print('The picture shows a ' + str(labels[str(pos_label)]))
