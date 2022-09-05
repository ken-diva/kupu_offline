import asyncio
import time
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/render', methods=['POST'])
async def render():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    await asyncio.sleep(10)
    return "uploaded!"


if __name__ == '__main__':
    app.run(debug=True)
