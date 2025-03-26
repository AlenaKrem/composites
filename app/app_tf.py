import flask
from flask import render_template
import tensorflow as tf
import torch
import numpy as np

print("Hello flask")

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':

        loaded_model = tf.keras.models.load_model('tf_model.tf')

        plt = float(flask.request.form['Плотность, кг/м3'])
        upr = float(flask.request.form['Mодуль упругости, ГПа'])
        otv = float(flask.request.form['Количество отвердителя, м.%'])
        epo = float(flask.request.form['Содержание эпоксидных групп,%_2'])
        tem = float(flask.request.form['Температура вспышки, С_2'])
        pov = float(flask.request.form['Поверхностная плотность, г/м2'])
        mod = float(flask.request.form['Модуль упругости при растяжении, ГПа'])
        pro = float(flask.request.form['Прочность при растяжении, МПа'])
        smo = float(flask.request.form['Потребление смолы, г/м2'])
        sha = float(flask.request.form['Шаг нашивки'])
        plo = float(flask.request.form['Плотность нашивки'])
        ugo0 = float(flask.request.form['Угол нашивки, град 0'])
        ugo90 = float(flask.request.form['Угол нашивки, град 90'])

        y_pred = loaded_model.predict(np.array([plt, upr, otv, epo, tem, pov, mod, pro, smo, sha, plo, ugo0, ugo90]).reshape(1,-1))
        return render_template('main.html', result = y_pred.item())

if __name__ == '__main__':
    app.run()