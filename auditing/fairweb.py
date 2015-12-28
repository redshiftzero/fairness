from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import os
import sys
from werkzeug import secure_filename
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
from fairness import metrics

from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8


colors = {
    'Black': '#000000',
    'Red':   '#FF0000',
    'Green': '#00FF00',
    'Blue':  '#0000FF',
}

upload_folder = 'uploads/'
allowed_extensions = set(['csv'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit
app._static_folder = ''


def getitem(obj, item, default):
    if item not in obj:
        return default
    else:
        return obj[item]


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in allowed_extensions


@app.route('/')
@app.route('/home')
def intro_page():
    return render_template('index.html')


@app.route('/about')
def about_page():
    return render_template('about.html')


@app.route('/transparency')
def practices_page():
    return render_template('transparency.html')


#@app.route('/upload/<filename>')
#def uploaded_file(filename):
#    rawcsv = '{}/{}'.format(app.config['UPLOAD_FOLDER'], filename)
#    df = pd.read_csv(rawcsv)
#    return render_template('upload.html', df=df)


@app.route('/fairness', methods=['GET', 'POST'])
def fairness_landing():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('do_audit',
                                    filename=filename))
    return render_template('fairness.html')


@app.route('/audit/<filename>')
def do_audit(filename):
    rawcsv = '{}/{}'.format(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(rawcsv)

    audit = {}
    audit['race_gf'] = metrics.group_fairness(df['race'].values, df['prediction'].values)
    audit['race_num_majority'] = len(df[df['race'] == 1])
    audit['race_num_minority'] = len(df[df['race'] == 0])

    df_pos = df[df['prediction'] == 1]
    audit['race_num_minority_pos'] = len(df_pos[df_pos['race'] == 0])
    audit['race_num_majority_pos'] = len(df_pos[df_pos['race'] == 1])

    audit['gender_num_majority'] = len(df[df['gender'] == 1])
    audit['gender_num_minority'] = len(df[df['gender'] == 0])
    audit['gender_num_minority_pos'] = len(df_pos[df_pos['gender'] == 0])
    audit['gender_num_majority_pos'] = len(df_pos[df_pos['gender'] == 1])

    audit['rel_race_error_rate'], audit['rel_pos_race_error_rate'], audit['rel_neg_race_error_rate'] = metrics.rel_error_rate(df['race'].values, df['prediction'].values, df['truth'].values)
    audit['rel_gender_error_rate'], audit['rel_pos_gender_error_rate'], audit['rel_neg_gender_error_rate'] = metrics.rel_error_rate(df['gender'].values, df['prediction'].values, df['truth'].values)
    audit['gender_gf'] = metrics.group_fairness(df['gender'].values, df['prediction'].values)

    df_pos = df[df['truth'] == 1]
    df['error'] = np.abs(df_pos['prediction'] - df_pos['truth'])
    df_incorrect = df[df['error'] > 0]
    audit['race_majority_wrong'] = float(len(df_incorrect['race'] == 1)) / float(len(df_pos))
    audit['race_minority_wrong'] = float(len(df_incorrect['race'] == 0)) / float(len(df_pos))
    audit['gender_majority_wrong'] = float(len(df_incorrect['gender'] == 1)) / float(len(df_pos))
    audit['gender_minority_wrong'] = float(len(df_incorrect['gender'] == 0)) / float(len(df_pos))

    return render_template('auditresult.html', df=df, filename=filename, audit=audit)


if __name__ == '__main__':
    # For security reasons, DO NOT ENABLE debugging 
    # on production machines
    app.debug = False

    app.run()