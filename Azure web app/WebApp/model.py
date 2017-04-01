import os
import sys
import scipy.misc
import numpy as np
import transform
import tensorflow as tf
from flask import Flask, request, jsonify, send_file, render_template
from io import BytesIO
from PIL import Image, ImageOps
import base64
import urllib

app = Flask(__name__)

models_dict = {
    'la_muse':'Model/la_muse.ckpt', 
    'scream':'Model/scream.ckpt',
    'wave':'Model/wave.ckpt',
    'rain_princess':'Model/rain_princess.ckpt',
    'udnie':'Model/udnie.ckpt',
    'wreck':'Model/wreck.ckpt'
    }

@app.route("/")
def index():
	return render_template('index.html')

@app.route("/uploader", methods=['POST'])
def upload_file():
    # Check valid image and compress
    try:
        img = Image.open(BytesIO(request.files['imagefile'].read())).convert('RGB')
        img = ImageOps.fit(img, (400, 400), Image.ANTIALIAS)
    except:
        error_msg = "Please choose an image file"
        return render_template('index.html', **locals())

    out_img = rundeeplearning(np.array(img), "Model/la_muse.ckpt")
    img_io = BytesIO()
    out_img.save(img_io, 'PNG')
    # Display re-sized picture
    png_output = base64.b64encode(img_io.getvalue())
    processed_file = urllib.parse.quote(png_output)
    return render_template('index.html', **locals())

@app.route("/api/styles", methods=['GET'])
def return_styles():
    return jsonify(models_dict)

@app.route("/api/uploader/<style>", methods=['POST'])
def api_upload_file(style):
    img = Image.open(BytesIO(request.files['imagefile'].read())).convert('RGB')
    if min(img.size) > 400:
        img = ImageOps.fit(img, (400, 400), Image.ANTIALIAS)
    in_img = np.array(img)
    model_file = models_dict[style]
    out_img =  rundeeplearning(in_img, model_file)
    return send_pil(out_img)

### Helpers utility functions
def send_pil(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=90)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    scale = float(style_scale)
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = _get_img(style_path, img_size=new_shape)
    return style_target

def get_img(src, img_size=False):
   img = src  # Already pill image
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img

## Evaluate image
def rundeeplearning(data_in, checkpoint_dir, device_t='/cpu:0', batch_size=1):
    img_shape = get_img(data_in).shape
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')
        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        # Load 
        saver.restore(sess, checkpoint_dir)
        X = np.zeros(batch_shape, dtype=np.float32)
        img = get_img(data_in)
        assert img.shape == img_shape
        X[0] = img
        _preds = sess.run(preds, feed_dict={img_placeholder:X})
        out_img = np.clip(_preds[0], 0, 255).astype(np.uint8)
        return Image.fromarray(out_img)

