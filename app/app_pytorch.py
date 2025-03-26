import flask
from flask import render_template
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print("Hello flask")

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':
        
        data_1 = pd.read_excel('./dataset/X_bp.xlsx', index_col=0)
        data_2 = pd.read_excel('./dataset/X_nup.xlsx', index_col=0)

        dataset = data_1.join(data_2, how='inner')
        dataset = pd.get_dummies(dataset, columns=['Угол нашивки, град'], dtype=int) 

        train_df, test_df = train_test_split(dataset.astype(np.float32), test_size=0.3, random_state=42, shuffle=True)
        transformer_train = Pipeline(steps=[('st_scaler',StandardScaler()), ('mm_scaler', MinMaxScaler()) ])
        transformer_test = Pipeline(steps=[('st_scaler',StandardScaler()), ('mm_scaler', MinMaxScaler()) ])

        preprocessor_train = transformer_train.fit(train_df.iloc[:,1:])
        preprocessor_test = transformer_test.fit(test_df.iloc[:,:1])

        loaded_model = torch.jit.load('torch_model.pt')

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

        preprocessed = preprocessor_train.transform((np.array([plt, upr, otv, epo, tem, pov, mod, pro, smo, sha, plo, ugo0, ugo90]).reshape(1,-1).astype('float32')))
        y_pred_raw = loaded_model(torch.tensor(preprocessed))
        y_pred = preprocessor_test.inverse_transform(y_pred_raw.detach().numpy())
        return render_template('main.html', result = y_pred.item())

if __name__ == '__main__':
    app.run()