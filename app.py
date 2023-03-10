from flask import Flask, render_template, request, redirect, url_for
from PIL import Image

import os, uuid, json
import module.my_engine_modified as engine
import module.decode as decoder

app = Flask(__name__)

# config for upload image
app.config['UPLOAD_FOLDER'] = 'upload/img/'

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/upload_img', methods=["POST"])
def upload_img():
  # upload front img file
  random_filename = str(uuid.uuid4())
  img_front = request.files['img_front']
  img_front_filename = 'img_front' + '.' + img_front.filename.split('.')[1]
  img_front.save(os.path.join(app.config['UPLOAD_FOLDER'], img_front_filename))

  #upload back img file
  random_filename = str(uuid.uuid4())
  img_back = request.files['img_back']
  img_back_filename = 'img_back' + '.' + img_back.filename.split('.')[1]
  img_back.save(os.path.join(app.config['UPLOAD_FOLDER'], img_back_filename))

  engine.render_data(img_front_filename, img_back_filename)
  return render_template('result.html')

@app.route('/edit_front')
def edit_front():
  return render_template('label_front.html')

@app.route('/edit_back')
def edit_back():
  return render_template('label_back.html')

@app.route('/export_front', methods=["POST"])
def export_front():
  jsonData = request.get_json()
  with open('./static/json_annotation/img_front.json', 'w') as f:
    json.dump(jsonData, f)
  return {
    'response' : 'export front success'
  }

@app.route('/export_back', methods=["POST"])
def export_back():
  jsonData = request.get_json()
  with open('./static/json_annotation/img_back.json', 'w') as f:
    json.dump(jsonData, f)
  return {
    'response' : 'export back success'
  }

@app.route('/decode')
def decode():

    decoder.convert_to_png('static\json_annotation\img_front.json', 'front')
    decoder.convert_to_png('static\json_annotation\img_back.json', 'back')

    return redirect(url_for('decode_result'))

@app.route('/decode_result')
def decode_result():
    # img = Image.open('upload/img/img_front.png')
    img = Image.open('static/img_front_result.png')
    # response = requests.get(decoded_front)
    # img_decoded = Image.open('upload\img\img_front.png')
    img_decoded = Image.open('image_front_done.png')

    # Pasting img2 image on top of img1 
    # starting at coordinates (0, 0)
    img.paste(img_decoded, (-35,8), mask=img_decoded)
    img.save("static/final_front.png")

    img = Image.open('upload/img/img_back.png')
    # response = requests.get(decoded_back)
    # img_decoded = Image.open('upload\img\img_back.png')
    img_decoded = Image.open('image_back_done.png')

    # Pasting img2 image on top of img1 
    # starting at coordinates (0, 0)
    img.paste(img_decoded, (-35,8), mask=img_decoded)
    img.save("static/final_back.png")

    return render_template("final_result.html")

@app.route('/coba')
def coba():
    return render_template('old/via_demo.html')

if __name__ == '__main__':
  app.run(debug=True)