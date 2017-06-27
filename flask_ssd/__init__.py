from flask import Flask, request, json, redirect, render_template, send_from_directory
from werkzeug.utils import secure_filename
import datetime
import urllib
import os
import pickle
import tensorflow as tf
import os
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from scipy.misc import imread

from flask_ssd.ssd_keras.ssd import SSD300
from flask_ssd.ssd_keras.ssd_utils import BBoxUtility
from PIL import Image, ImageDraw
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
app = Flask(__name__)

# global variable
webapp_path = '/var/www/flask_ssd/flask_ssd'
file_db_path = '/home/tangyu/ssd_db'
unknown_folder_path = os.path.join(file_db_path, 'unknown')
result_folder_path = os.path.join(file_db_path, 'result')
upload_folder_path = os.path.join(file_db_path, 'upload')
temp_folder_path = os.path.join(file_db_path, 'tmp')

# load logo classes file
brand_dict = pickle.load(open(os.path.join(webapp_path, 'brand_map.pkl'), 'rb'))
print(brand_dict)

NUM_CLASSES = len(brand_dict) + 1  # NUM_CLASS equal real class numbers + 1

# init a model
input_shape = (300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
chk_path = os.path.join(webapp_path, "checkpoints")

#load weights from *.hdf5 file
for f in os.listdir(chk_path):
    if ".hdf5" in f:
        model.load_weights(os.path.join(chk_path, f), by_name=True)
        break

bbox_util = BBoxUtility(NUM_CLASSES)

# define image generator
datagen = ImageDataGenerator(
    rotation_range=270,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

# get file basename and extension
def get_file_name_ext(filename):
    (filepath, tempfilename) = os.path.split(filename);
    (basename, extension) = os.path.splitext(tempfilename)
    return basename, extension


def get_image_class_from_local_file(filename):
    inputs = []
    images = []
    img = image.load_img(filename, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(filename))
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))
    preds = model.predict(inputs, batch_size=1)
    results = bbox_util.detection_out(preds)
    ret_dict = {"recResult": [], "message": "success"}
    if len(results[0]) < 1:
        return json.dumps(ret_dict)
    # Parse the outputs.
    print(results)
    det_label = results[0][:, 0]
    det_conf = results[0][:, 1]
    det_xmin = results[0][:, 2]
    det_ymin = results[0][:, 3]
    det_xmax = results[0][:, 4]
    det_ymax = results[0][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.4]


    if len(top_indices) > 0:
        rec_list = []
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        im = Image.open(filename)
        width_ratio = im.size[0] / 300
        height_ratio = im.size[1] / 300
        draw = ImageDraw.Draw(im)
        max_score = -1
        max_score_label = "unknown"
        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            # label_name = voc_classes[label - 1]
            if (label - 1) in brand_dict.keys():
                label_name = brand_dict[label - 1]
            else:
                label_name = label
            label_name = label_name.lower()
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            act_score = round(score * 100, 2)
            single_obj_dict = {"imageClass": label_name, "score": act_score, "tlX": xmin, "tlY": ymin,
                               "brX": xmax, "brY": ymax}
            rec_list.append(single_obj_dict)
            if act_score > max_score:
                max_score_label = label_name
                max_score = act_score
            # print(display_txt)
            draw.rectangle((xmin * width_ratio, ymin * height_ratio, xmax * width_ratio, ymax * height_ratio))
            draw.text((xmin * width_ratio, ymin * height_ratio), display_txt)

        filename = os.path.basename(filename)
        (shortname, extname) = get_file_name_ext(filename)
        im = im.convert('RGB')
        im.save(os.path.join(result_folder_path, max_score_label + "-" + shortname + ".jpg"))

        ret_dict["recResult"] = rec_list
    else:
        im = Image.open(filename)
        im = im.convert('RGB')
        (shortname, extname) = get_file_name_ext(filename)
        filename = os.path.join(unknown_folder_path, shortname + ".jpg")
        im.save(filename)
    return json.dumps(ret_dict)


@app.route("/")
def hello():
    return "Welcome to id-bear logo recognition system! NUM_CLASS={0}, {1}".format(NUM_CLASSES, brand_dict)


@app.route('/api/get_image_class_file', methods=['POST'])
def api_get_image_class_file():
    if "multipart/form-data" in request.headers['Content-Type']:
        # Get the name of the uploaded file
        file = request.files['file']
        # Check if the file is one of the allowed types/extensions
        if file:
            print(file.filename)
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            # Move the file form the temporal folder to
            # the upload folder we setup
            filename = datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'-'+filename
            image_path = os.path.join(upload_folder_path, filename)
            file.save(image_path)

            # force img to rgb format
            (shortname, extname) = get_file_name_ext(image_path)
            im = Image.open(image_path)
            im = im.convert('RGB')
            jpg_image_path = os.path.join(upload_folder_path, shortname + ".jpg")
            im.save(jpg_image_path)

            ret = get_image_class_from_local_file(jpg_image_path)
            # if upload picture is not jpg format, then convert its format to jpg

            if '.jpg' not in extname:
                os.remove(image_path)
            return ret
        else:
            ret_dict = {"message": "file is empty"}
            return json.dumps(ret_dict)
    else:
        ret_dict = {"message": "please use multipart/form-data in headers' Content-Type"}
        return json.dumps(ret_dict)


@app.route('/get_augment_result', methods=['GET', 'POST'])
def logo_upload():
    if request.method == 'GET':
        return render_template('get_augment_result.html')
    elif request.method == 'POST':
        f = request.files['file']
        if f:
            fname = f.filename
            file_full_path = os.path.join(temp_folder_path, fname)
            f.save(file_full_path)

            # uncompress
            unzip_cmd = 'unzip ' + file_full_path + ' -d '+temp_folder_path
            os.system(unzip_cmd)

            # augmentation

        else:
            return render_template('get_augment_result.html')
    return "upload successfully"


if __name__ == "__main__":
    app.run()
