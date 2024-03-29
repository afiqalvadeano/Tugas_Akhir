from flask import jsonify, render_template, request, flash, redirect, url_for
from app.models.Tahun import *
from app.models.Data import *
from app.models.Hasil import *
import io
import ee
import geemap
import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import earthpy.plot as ep
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Input, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical, plot_model, model_to_dot
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


def index():
    tahun = Tahun.get().serialize()
    return render_template('pages/data.html', tahun=tahun, segment='data')

def delete(id):
    tahun = Tahun.find(id).delete()
    data  = Data.where('tahun_id', id).delete()
    flash('Data berhasil dihapus.!', 'success')
    return redirect(url_for("data_index"))

def get_ndvi_data():
    np.random.seed(123)
    tf.random.set_seed(123)
    tf.keras.utils.set_random_seed(123)

    cek_tahun = Tahun.where('tahun', int(request.form['yearInput'])).first()
    if cek_tahun is not None:
        flash('Data pada tahun tersebut sudah tersedia.!', 'danger')
        return redirect(url_for('data_index'))

    ee.Authenticate()
    ee.Initialize(project='1031392701041')

    # Tentukan wilayah
    region_of_interest = ee.Geometry.Rectangle([110.0089, -8.2414, 110.8603, -7.4959]).buffer(5000)
    
    # Tentukan rentang waktu
    start_date = request.form['yearInput']+'-01-01'
    end_date   = request.form['yearInput']+'-12-31'

    # Dapatkan citra Sentinel-2 dan hitung NDVI
    image = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(region_of_interest) \
        .filterDate(start_date, end_date) \
        .median()

    # Hitung NDVI dari citra Sentinel-2
    red_band = image.select('B4')  # Pilih band merah (Red)
    nir_band = image.select('B8')  # Pilih band inframerah dekat (Near Infrared)
    ndvi = nir_band.subtract(red_band).divide(nir_band.add(red_band)).rename('NDVI')

    # Clip the NDVI image to the specified ROI
    ndvi_roi = ndvi.clip(region_of_interest)

    # Convert the NDVI image to a visualization-friendly format
    ndvi_vis = ndvi_roi.visualize(min=-1, max=1, palette=['blue', 'white', 'green'])

    # Save the NDVI image to a temporary file
    temp_file = 'static/ndvi_image/'+ request.form['yearInput'] +'.tif'
    geemap.ee_export_image(ndvi_vis, filename=temp_file, scale=100, region=region_of_interest, file_per_band=False)

    # Load image
    image_n = rasterio.open(temp_file)
    height = image_n.height
    width = image_n.width

    image_vis = []
    for x in range(1, 4):
        image_vis.append(image_n.read(x))
    image_vis = np.stack(image_vis)

    plot_size = (8, 8)
    fig, ax = plt.subplots(figsize=plot_size)
    ep.plot_rgb(
        image_vis,
        ax=ax,
        stretch=True,
    )
    image_name = 'static/ndvi_image/'+ request.form['yearInput'] +'.jpg'
    plt.savefig(image_name, format='jpg')
    plt.clf()
    plt.close()

    # Tambahkan NDVI ke citra
    ndvi_image = image.addBands(ndvi)

    ndvi_data = ndvi_image.select(['B4', 'B8', 'NDVI']).reduceRegion(
        reducer=ee.Reducer.toList(),
        geometry=region_of_interest,
        scale=1000,
    )

    # Konversi data ke Pandas DataFrame
    df = pd.DataFrame.from_dict(ndvi_data.getInfo())

    # Tentukan fungsi untuk menetapkan label berdasarkan rentang NDVI
    def determine_label(ndvi_value):
        if ndvi_value < 0:
            return 'air'
        elif ndvi_value >= 0 and ndvi_value <= 0.2:
            return 'lahan terbangun'
        else:
            return 'vegetasi'

    # Terapkan fungsi untuk menentukan label pada setiap baris DataFrame
    df['label'] = df['NDVI'].apply(determine_label)
    print(df.label.value_counts())

    # Simpan DataFrame ke file Excel
    # excel_filename = 'ndvi_data.csv'
    # df.to_csv(excel_filename, index=False)

    # Split data
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    df_train = X_train.copy()
    df_train['label'] = y_train

    df_test = X_test.copy()
    df_test['label'] = y_test

    # Label encoder
    le = LabelEncoder()
    le.fit(y_train)

    y_train = le.fit_transform(y_train)
    y_test  = le.fit_transform(y_test)

    # Convert samples dataframe (pandas) to numpy array
    train_input = reshape_input(np.array(X_train))
    test_input = reshape_input(np.array(X_test))

    # Also make label data to categorical
    train_output = to_categorical(y_train, len(le.classes_) + 1)
    test_output = to_categorical(y_test, len(le.classes_) + 1)

    # Show the data shape
    print(f'Train features: {train_input.shape}\nTest features: {test_input.shape}\nTrain label: {train_output.shape}\nTest label: {test_output.shape}')

    # Make model for our data
    # Input shape
    train_shape = train_input.shape
    input_shape = (train_shape[1], train_shape[2])

    # Model parameter
    neuron = 128
    drop = 0.2
    kernel = 1
    pool = 1

    # Make sequential model
    model = Sequential([
    Input(input_shape),
    Conv1D(neuron * 1, kernel, activation='relu'),
    Conv1D(neuron * 1, kernel, activation='relu'),
    MaxPooling1D(pool),
    Dropout(drop),
    Conv1D(neuron * 2, kernel, activation='relu'),
    Conv1D(neuron * 2, kernel, activation='relu'),
    MaxPooling1D(pool),
    Dropout(drop),
    GlobalMaxPooling1D(),
    Dense(neuron * 2, activation='relu'),
    Dropout(drop),
    Dense(neuron * 1, activation='relu'),
    Dropout(drop),
    Dense(len(le.classes_) + 1, activation='softmax')
    ])

    # Train the model
    # Compline the model
    model.compile(
        optimizer='Adam',
        loss='CategoricalCrossentropy',
        metrics=['accuracy']
    )

    # Create callback to stop training if loss not decreasing
    stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5
    )

    # Fit the model
    history = model.fit(
        x=train_input, y=train_output,
        validation_data=(test_input, test_output),
        batch_size=512,
        callbacks=[stop],
        epochs=100,
    )

    # Save model CNN for later use
    model.save('static/model/cnn_model_'+ request.form['yearInput'] +'.h5')

    tahun = Tahun()
    tahun.tahun = request.form['yearInput']
    tahun.save()

    for index, row in df.fillna(0).iterrows():
        tmp_store = {
            'tahun_id': tahun.serialize()['id'],
            'B4'      : row['B4'],
            'B8'      : row['B8'],
            'ndvi'    : row['NDVI'],
            'label'   : row['label']
        }
        Data.insert(tmp_store)

    new_input  = reshape_input(np.array(X))
    prediction = np.argmax(model.predict(new_input), 1).flatten()

    df['klasifikasi'] = le.inverse_transform(prediction)

    # Hitung jumlah piksel untuk setiap kelas dari hasil klasifikasi
    n_pixels_air = df[df['klasifikasi'] == 'air'].shape[0]
    n_pixels_lahan_terbangun = df[df['klasifikasi'] == 'lahan terbangun'].shape[0]
    n_pixels_vegetasi = df[df['klasifikasi'] == 'vegetasi'].shape[0]

    # Resolusi piksel (dalam meter persegi)
    resolution = 1000

    # Konversi jumlah piksel menjadi luas dalam meter persegi
    area_air = (n_pixels_air * resolution) / 1000000
    area_lahan_terbangun = (n_pixels_lahan_terbangun * resolution) / 1000000
    area_vegetasi = (n_pixels_vegetasi * resolution) / 1000000

    # Print hasil luas
    print("Luas Air:", area_air, "km2")
    print("Luas Lahan Terbangun:", area_lahan_terbangun, "km2")
    print("Luas Vegetasi:", area_vegetasi, "km2")

    hasil = Hasil()
    hasil.tahun_id = tahun.serialize()['id']
    hasil.luas_air = area_air
    hasil.luas_lahan_terbangun = area_lahan_terbangun
    hasil.luas_vegetasi = area_vegetasi
    hasil.save()

    flash('Proses training selesai & data berhasil disimpan.!', 'success')
    return redirect(url_for('data_index'))

def detail_data(id):
    tahun = Tahun.where('id', id).first().serialize()
    data  = Data.where('tahun_id', tahun['id']).select('tahun_id', 'B4', 'B8', 'ndvi', 'label').get().serialize()

    df_luas = Hasil.where('tahun_id', tahun['id']).get().serialize()

    df = pd.DataFrame(data)
    df = df.drop('tahun_id', axis=1)

    # Split data
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    df_train = X_train.copy()
    df_train['label'] = y_train

    df_test = X_test.copy()
    df_test['label'] = y_test

    # Label encoder
    le = LabelEncoder()
    le.fit(y_train)

    y_train = le.fit_transform(y_train)
    y_test  = le.fit_transform(y_test)

    # Convert samples dataframe (pandas) to numpy array
    train_input = reshape_input(np.array(X_train))
    test_input = reshape_input(np.array(X_test))

    # Also make label data to categorical
    train_output = to_categorical(y_train, len(le.classes_) + 1)
    test_output = to_categorical(y_test, len(le.classes_) + 1)

    # Show the data shape
    print(f'Train features: {train_input.shape}\nTest features: {test_input.shape}\nTrain label: {train_output.shape}\nTest label: {test_output.shape}')

    n_model = load_model('static/model/cnn_model_'+ str(tahun['tahun']) +'.h5')

    # Predict test data
    prediction = np.argmax(n_model.predict(test_input), 1).flatten()
    label      = np.argmax(test_output, 1).flatten()

    df_test['klasifikasi'] = le.inverse_transform(prediction)

    cm           = confusion_matrix(label, prediction)
    CLASSES      = ['Air', 'Lahan Terbangun', 'Vegetasi']
    df_confusion = pd.DataFrame(cm, index = CLASSES, columns = CLASSES)

    sns.heatmap(df_confusion, annot=True, fmt = "d", cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Prediction Label')
    plt.ylabel('Actual Label')
    plt.savefig('static/cm_'+ str(tahun['tahun']) + '.jpg')
    plt.close()

    accuracy_  = round(accuracy_score(label, prediction)*100, 2)
    precision_ = round(precision_score(label, prediction, average='weighted')*100, 2)
    recall_    = round(recall_score(label, prediction, average='weighted')*100, 2)
    f1_score_  = round(f1_score(label, prediction, average='weighted')*100, 2)

    print('Accuracy  : ', accuracy_)
    print('Precision : ', precision_)
    print('Recall    : ', recall_)
    print('F1-score  : ', f1_score_)

    return render_template('pages/detail_data.html', data=data, df_train=df_train, df_test=df_test,
    accuracy_=accuracy_, precision_=precision_, recall_=recall_, f1_score_=f1_score_, tahun=tahun['tahun'],
    df_luas=df_luas, segment='data')

# Function to reshape array input
def reshape_input(array):
    shape = array.shape
    return array.reshape(shape[0], shape[1], 1)