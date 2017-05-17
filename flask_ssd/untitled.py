from flask import Flask, request, json, redirect, render_template, send_from_directory
from werkzeug.utils import secure_filename
import datetime
import urllib
import os
import pickle
import tensorflow as tf
_ = tf.contrib.tensor_forest
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from scipy.misc import imread

from ssd_keras.ssd import SSD300
from ssd_keras.ssd_utils import BBoxUtility
from PIL import Image, ImageDraw
import zipfile
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.getcwd()+'\\uploads'

unknown_folderpath = os.getcwd()+'\\unknown'
result_folderpath = os.getcwd()+'\\result'
temp_filepath = os.getcwd()+'\\tmp'

# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
brand_dict = pickle.load(open('brand_map.pkl', 'rb'))
print(brand_dict)
#NUM_CLASSES = 27
#NUM_CLASSES = 30 # actual class + 1
NUM_CLASSES = len(brand_dict) + 1

input_shape = (300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
chk_path = "./checkpoints"
# model.load_weights('weights_SSD300.hdf5', by_name=True)
#model.load_weights('./checkpoints/weights.29-4.12.hdf5', by_name=True)
#model.load_weights('./checkpoints/weights.19-1.97.hdf5', by_name=True)
#model.load_weights('./checkpoints/weights.29-2.15.hdf5', by_name=True)
#model.load_weights('./checkpoints/weights.29-1.83.hdf5', by_name=True)
for f in os.listdir(chk_path):
    if ".hdf5" in f:
        model.load_weights(os.path.join(chk_path, f), by_name=True)



bbox_util = BBoxUtility(NUM_CLASSES)

def get_file_name_ext(filename):
    (filepath, tempfilename) = os.path.split(filename);
    (shotname, extension) = os.path.splitext(tempfilename)
    return shotname,extension

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
    

    # Parse the outputs.
    det_label = results[0][:, 0]
    det_conf = results[0][:, 1]
    det_xmin = results[0][:, 2]
    det_ymin = results[0][:, 3]
    det_xmax = results[0][:, 4]
    det_ymax = results[0][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]
    ret_dict = {"recResult": [], "message": "success"}


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
            #print(display_txt)
            draw.rectangle((xmin * width_ratio, ymin * height_ratio, xmax * width_ratio, ymax * height_ratio))
            draw.text((xmin * width_ratio, ymin * height_ratio), display_txt)

        filename = os.path.basename(filename)
        (shortname, extname) = get_file_name_ext(filename)
        im = im.convert('RGB')
        im.save(os.path.join(result_folderpath, max_score_label+"-"+shortname + ".jpg"))
            
        ret_dict["recResult"] = rec_list
    else:
        im = Image.open(filename)
        (shortname, extname) = get_file_name_ext(filename)

        filename = unknown_folderpath + "\\" + shortname + ".jpg"

        im.save(filename)
    return json.dumps(ret_dict)

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['GET','POST'])
def logo_upload():
    if request.method == 'GET':
        return render_template('upload.html')
    elif request.method == 'POST':
        f = request.files['file']
        fname = f.filename
        file_full_path = os.path.join(temp_filepath, fname)
        f.save(file_full_path)
        
        # uncompress
        zfile = zipfile.ZipFile(file_full_path, 'r') 
        for p in zfile.namelist():
            if p.endswith('/'):
                full_p = os.path.join(temp_filepath, p)
                if not os.path.exists(full_p):
                    os.mkdir(full_p)
            else:
                full_p = os.path.join(temp_filepath, p)
                open(full_p, 'wb').write(zfile.read(full_p)) 
        return "Upload OK"
        
        
           
@app.route('/')
def hello_world():
    return redirect("https://www.showdoc.cc/15504?page_id=140847")


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
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filename)
            ret = get_image_class_from_local_file(filename)
            # if upload picture is not jpg format, then convert its format to jpg
            (shortname, extname) = get_file_name_ext(filename)

            if extname != ".jpg":
                im = Image.open(filename)
                im = im.convert('RGB')
                im.save(os.path.join(app.config['UPLOAD_FOLDER'], shortname + ".jpg"))
                os.remove(filename)
            return ret
        else:
            ret_dict = {"message": "file is empty"}
            return json.dumps(ret_dict)
    else:
        ret_dict = {"message": "please use multipart/form-data in headers' Content-Type"}
        return json.dumps(ret_dict)


@app.route('/api/get_image_class', methods=['POST'])
def api_get_image_class():
    image_url = request.form['imageUrl']
    filename = os.path.basename(image_url)
    urllib.request.urlretrieve(image_url, filename)
    ret = get_image_class_from_local_file(filename)
    return ret


@app.route('/api/get_image_class_json', methods=['POST'])
def api_get_image_class_json():
    if request.headers['Content-Type'] == 'text/plain':
        ret_dict = {"message": "please set the correct Content-Type"}
        return json.dumps(ret_dict)
    elif request.headers['Content-Type'] == 'application/json':
        if request.is_json:  # if pass json from caller
            req_dict = eval(json.dumps(request.json))
            if('imageUrl' in req_dict):  # start process image
                filename = os.path.basename(req_dict['imageUrl'] )
                urllib.request.urlretrieve(req_dict['imageUrl'], filename)
                ret = get_image_class_from_local_file(filename)
                return ret
            else:
                ret_dict = {"recResult": [], "message": "you pass a wrong key"}
                return json.dumps(ret_dict)
        else:
            ret_dict = {"recResult": [], "message": "please use json format"}
            return json.dumps(ret_dict)
    else:
        ret_dict = {"recResult": [], "message": "please use json format"}
        return json.dumps(ret_dict)


if __name__ == '__main__':
    app.run()
