from flask import Flask, render_template, request

import os, uuid, json
import module.my_engine_modified as engine

app = Flask(__name__)

# config for upload image
app.config['UPLOAD_FOLDER'] = 'upload/img/'

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/render', methods=["POST"])
def render():
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
  return render_template('render.html')

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
  print(jsonData)
  return {
    'response' : 'export front success'
  }

@app.route('/export_back', methods=["POST"])
def export_back():
  jsonData = request.get_json()
  with open('./static/json_annotation/img_back.json', 'w') as f:
    json.dump(jsonData, f)
  print(jsonData)
  return {
    'response' : 'export back success'
  }

@app.route('/coba')
def coba():
    return render_template('old/via_demo.html')

if __name__ == '__main__':
  app.run(debug=True)