#!/usr/bin/env python
# coding: utf-8

# In[2]:


#precisa instala alguns pacotes.

#NO MAC
'''
brew install tcl-tk
pip install lxml
brew install protobuf
brew install pil




'''
'''
# For CPU
!pip install tensorflow
# For GPU
!pip install tensorflow-gpu
!pip install --user Cython
!pip install --user contextlib2
!pip install lxml
!pip install PIL

!pip install --user Cython
!pip install --user contextlib2
!pip install --user pillow
!pip install --user lxml



#Precisa configurar o COCOAPI
!git clone https://github.com/cocodataset/cocoapi.git


#parece que ele precisa links direto para o folder onde está as funcoes da COCOAPI
!cp object_detection_tutorial.ipynb  /Users/arnaldoljr/.virtualenvs/cv/lib/python3.7/site-packages/tensorflow/models/research/object_detection/


OBSERVACAO

Any model exported using the export_inference_graph.py tool can be loaded here simply 
by changing PATH_TO_FROZEN_GRAPH to point to a new .pb file.
By default we use an "SSD with Mobilenet" model here. See the detection model zoo
for a list of other models that can be run out-of-the-box with varying speeds and accuracies.



'''
# This is needed to display the images.
get_ipython().run_line_magic('matplotlib', 'inline')


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2


from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#tem que carregar da pasta correta...

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util

from utils import visualization_utils as vis_util




    # What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
        
        
        
#MÉTODO otimizado de carregar o tensorflow sem muito custo...
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
#fazer o mapemento do label que ele dedecta para o que realmente é....tipo...label 5 quer dizer um 'carro' por exemplo

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


#funcao para otimizar o a entrada..
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#funcao para puxar a webcam

# Using OpenCV to initialize the webcam
cap = cv2.VideoCapture(0)

with detection_graph.as_default():
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = False
    with tf.Session(graph=detection_graph, config=config) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while True:
            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=10)
            
            cv2.imshow('MobileNet SSD - Object Detection', image_np)
            if cv2.waitKey(1) == 13: #13 is the Enter Key
                break
            
# Release camera and close windows
cap.release()
cv2.destroyAllWindows()  


# In[ ]:




