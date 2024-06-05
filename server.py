from flask import Flask, request
from app.controllers import home, data, luas
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = False

@app.route("/")
def index():
    return home.index()

@app.route('/index_data')
def data_index():
    return data.index()

@app.route("/data/<int:id>/delete")
def data_delete(id):
    return data.delete(id)

@app.route('/data/get_ndvi_data', methods=['POST'])
def get_data():
    return data.get_ndvi_data()

@app.route("/data/detail/<int:id>")
def detail_index(id):
    return data.detail_data(id)

@app.route('/index_luas')
def luas_index():
    return luas.index()

app.secret_key = '3RDLwwtFttGSxkaDHyFTmvGytBJ2MxWT8ynWm2y79G8jm9ugYxFFDPdHcBBnHp6E'
app.config['SESSION_TYPE'] = 'filesystem'

@app.context_processor
def inject_stage_and_region():
    return dict(APP_NAME=os.environ.get("APP_NAME"),
        APP_AUTHOR=os.environ.get("APP_AUTHOR"),
        APP_TITLE=os.environ.get("APP_TITLE"),
        APP_LOGO=os.environ.get("APP_LOGO"))

if __name__ == "__main__":
    app.run()
    # app.run(host='0.0.0.0', port=5591)