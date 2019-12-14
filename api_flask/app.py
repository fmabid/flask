import glob
import logging
import os

from flask import Flask, send_from_directory, request, jsonify
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras import backend as K
from keras.models import load_model
import numpy as np


app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.logger.info(PROJECT_HOME)

upload_dir = app.config['UPLOAD_FOLDER']


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


def delete_image(img_name):
    os.remove(img_name)


@app.route('/', methods=['POST'])
def post_img():

    if request.method == 'POST' and request.files['image']:
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        img_name = secure_filename(img.filename)
        create_new_folder(format(app.config['UPLOAD_FOLDER']))
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)
        return 'Uploaded {}'.format(img_name)
    else:
        return "Where is the image?"


@app.route('/', methods=['GET'])
def get_img():
    app.logger.info(PROJECT_HOME)
    list_of_files = glob.glob('uploads/*')
    img_name = list_of_files[0].replace('uploads/', '')
    #print('------>  ' +img_name)

    if request.method == 'GET':
        app.logger.info(app.config['UPLOAD_FOLDER'])
        try:
            #Before prediction
            K.clear_session()
            # Give the path of your .h5 file here
            wd = os.getcwd()
            model_path = wd + '/mobilenetv2.h5'
            model = load_model(model_path)
            # Give the path of your resized image here (224X224X3 size)
            # Use a good resizer from Java compatible with Android before sending to the server for disease detection
            # Your resized image should retain good quality
            img_path = wd + '/uploads/'+img_name

            img = image.load_img(img_path, target_size=(224, 224, 3))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.
            index_disease_dict = {0: 'BLB', 1: 'BPH', 2: 'Brown_Spot',
                                  3: 'False_Smut', 4: 'Healthy_Plant', 5: 'Hispa',
                                  6: 'Neck_Blast', 7: 'Sheath_Blight_Rot', 8: 'Stemborer'}
            images = np.vstack([img_tensor])
            #classes = model.predict_classes(images, batch_size=10)
            classes = model.predict(img_tensor)
            y_pred = np.argmax(classes[0])
            # The following variable contains the string that you will be sending back to the mobile from server
            #print(index_disease_dict[y_pred])

            # return send_from_directory(app.config['UPLOAD_FOLDER'], img_name, as_attachment=True)
            return jsonify({"diseaseName": index_disease_dict[y_pred]})
        finally:
            app.logger.info('Deleting {}'.format(img_name))
            delete_image(list_of_files[0])
            #Before prediction
            K.clear_session()



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
