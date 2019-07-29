
# coding: utf-8

# In[1]:


# testes com as redes neurais VGG 15 , 16 / inception v3 e Resnet

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing import image
import cv2
from os import listdir
from os.path import isfile, join
import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50
 
resnet_model = ResNet50(weights='imagenet')



img_path = './images/dog.jpg' 

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = resnet_model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])



# Our openCV function that displays the image and it's predicted labels 
def draw_test(name, preditions, input_im):
    """Function displays the output of the prediction alongside the orignal image"""
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, imageL.shape[1]+300 ,cv2.BORDER_CONSTANT,value=BLACK)
    img_width = input_im.shape[1]
    for (i,predition) in enumerate(preditions):
        string = str(predition[1]) + " " + str(predition[2])
        cv2.putText(expanded_image,str(name),(img_width + 50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),1)
        cv2.putText(expanded_image,string,(img_width + 50,50+((i+1)*50)),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1)
    cv2.imshow(name, expanded_image)
print("\n\n........\n\n")
print("Teste com redes neurais VGG16 e Inception")
# Get images located in ./images folder    
mypath = "./images/"
file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Loop through images run them through our classifer
for file in file_names:

    from keras.preprocessing import image # Need to reload as opencv2 seems to have a conflict
    img = image.load_img(mypath+file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    #load image using opencv
    img2 = cv2.imread(mypath+file)
    imageL = cv2.resize(img2, None, fx=.5, fy=.5, interpolation = cv2.INTER_CUBIC) 
    
    # Get Predictions
    preds = resnet_model.predict(x)
    preditions = decode_predictions(preds, top=3)[0]
    draw_test("Predictions", preditions, imageL) 
    cv2.waitKey(0)

cv2.destroyAllWindows()

#Loads the VGG16 model
vgg_model = vgg16.VGG16(weights='imagenet')
 
# Loads the Inception_V3 model
inception_model = inception_v3.InceptionV3(weights='imagenet')
 
# Loads the ResNet50 model 
# uncomment the line below if you didn't load resnet50 beforehand
#resnet_model = resnet50.ResNet50(weights='imagenet')

print("\n.........\n\n")
print("Comparativo com todas as redes....")


# Get images located in ./images folder    
mypath = "./images/"
file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Loop through images run them through our classifer
for file in file_names:

    from keras.preprocessing import image # Need to reload as opencv2 seems to have a conflict
    img = image.load_img(mypath+file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    #load image using opencv
    img2 = cv2.imread(mypath+file)
    imageL = cv2.resize(img2, None, fx=.5, fy=.5, interpolation = cv2.INTER_CUBIC) 
    
    # Get VGG16 Predictions
    preds_vgg_model = vgg_model.predict(x)
    preditions_vgg = decode_predictions(preds_vgg_model, top=3)[0]
    draw_test("VGG16 Predictions", preditions_vgg, imageL) 
    
    # Get Inception_V3 Predictions
    preds_inception = inception_model.predict(x)
    preditions_inception = decode_predictions(preds_inception, top=3)[0]
    draw_test("Inception_V3 Predictions", preditions_inception, imageL) 

    # Get ResNet50 Predictions
    preds_resnet = resnet_model.predict(x)
    preditions_resnet = decode_predictions(preds_resnet, top=3)[0]
    draw_test("ResNet50 Predictions", preditions_resnet, imageL) 
    
    cv2.waitKey(0)

cv2.destroyAllWindows()

