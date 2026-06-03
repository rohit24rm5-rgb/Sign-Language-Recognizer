import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from cvzone.ClassificationModule import Classifier

class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

tf.keras.utils.get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})

classifier = Classifier(
    "C:/Users/rohit/OneDrive/Desktop/converted_keras/keras_model.h5",
    "C:/Users/rohit/OneDrive/Desktop/converted_keras/labels.txt"
)
print("Model loaded successfully!")
