# Make sure that Anaconda, tensorflow and keras are installed in your server
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

import os


# Give the path of your .h5 file here
wd = os.getcwd()
model_path = wd + '/model_stage2.h5'
model = load_model(model_path)
# Give the path of your resized image here (224X224X3 size)
# Use a good resizer from Java compatible with Android before sending to the server for disease detection
# Your resized image should retain good quality 
img_path = wd + '/Testable_resized_image/FS.jpeg'

img = image.load_img(img_path, target_size=(224, 224, 3))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
index_disease_dict = {0:'BLB', 1:'BPH', 2:'Brown_Spot', 
                3:'False_Smut', 4:'Healthy_Plant', 5:'Hispa',
                6:'Neck_Blast', 7:'Sheath_Blight_Rot', 8:'Stemborer'}
images = np.vstack([img_tensor])
classes = model.predict_classes(images, batch_size=10)
# The following variable contains the string that you will be sending back to the mobile from server
print(index_disease_dict[classes[0]])
