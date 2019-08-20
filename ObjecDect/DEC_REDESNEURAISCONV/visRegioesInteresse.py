# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QbzEh49cNr_QiLnd656oBaA4grWFBhzb
"""

#idenficando regioes de interesse ....mapear com a rede neural e depois visualizar com mapas de calor..

import cv2
from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions


#usar a VGG16...pré treinanda e com boa acurácia...
model = VGG16(weights='imagenet')


from google.colab import drive
drive.mount('/content/drive')



img_path = '/content/dog.jpeg'


img1 = image.load_img(img_path)
plt.imshow(img1);

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print("\n.....")
print('\nClassificando  a imgem de entrada....')

#ele identifica a raça do cachorro...
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])


#puxar o indice da classe..
np.argmax(preds[0])

tiger_output = model.output[:, 292]
last_conv_layer = model.get_layer('block5_conv3')

# Gradients of the Tiger class wrt to the block5_conv3 filer
grads = K.gradients(tiger_output, last_conv_layer.output)[0]

# Each entry is the mean intensity of the gradient over a specific feature-map channel 
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# Accesses the values we just defined given our sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# Values of pooled_grads_value, conv_layer_output_value given our input image
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature-map array by the 'importance' 
# of this channel regarding the input image 
for i in range(512):
    #channel-wise mean of the resulting feature map is the Heatmap of the CAM
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

plt.savefig("regioesDestaque.png")


#visualizar dentro da imagem


img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img

save_img_path = '/content/cachorro_cam.jpg'

cv2.imwrite(save_img_path, superimposed_img)

img1 = image.load_img(save_img_path)
plt.imshow(img1)
plt.savefig("imagDestaque.png")