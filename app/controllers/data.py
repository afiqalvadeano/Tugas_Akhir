from flask import render_template, request, flash, redirect, url_for
from app.models.Tahun import *
from app.models.Hasil import *
import ee
import geemap
import rasterio
import earthpy.plot as ep
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')

# Parameter
CLASSES     = ['Air', 'Lahan Terbangun', 'Vegetasi']
N_CLASSES   = len(CLASSES)
PALETTE     = ['#87CEFA', '#F08080', '#90EE90']

model_path = 'static/model/cnn_model.h5'

def index():
    tahun = Tahun.order_by('tahun', 'asc').get().serialize()
    return render_template('pages/data.html', tahun=tahun, segment='data')

def delete(id):
    tahun = Tahun.find(id).delete()
    flash('Data berhasil dihapus.!', 'success')
    return redirect(url_for("data_index"))

# Function to mask clouds using the Sentinel-2 QA60 band
def maskS2clouds(image):
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000)

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

    # Tentukan rentang waktu
    start_date = request.form['yearInput']+'-01-01'
    end_date   = request.form['yearInput']+'-12-31'

    # Define the region of interest (ROI)
    roi = ee.Geometry.Rectangle([110.0089, -8.2414, 110.8603, -7.4959])

    # Create an image collection for Sentinel-2
    sentinel2 = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(roi) \
        .filterDate(ee.Date(start_date), ee.Date(end_date)) \
        .map(maskS2clouds) \
        .median()

    # Calculate NDVI
    ndvi_ = sentinel2.normalizedDifference(['B8', 'B4']).rename('NDVI')  # Redefine to 'NDVI'

    # Combine Sentinel-2 image with NDVI
    sentinel2_with_ndvi = sentinel2.addBands(ndvi_)

    # Save the NDVI image to a temporary file
    temp_file = 'static/ndvi_image/'+ request.form['yearInput'] +'.tif'
    scale = 200
    geemap.ee_export_image(sentinel2_with_ndvi, filename=temp_file, scale=scale, region=roi)

    # Load image
    image_n = rasterio.open(temp_file)
    height = image_n.height
    width = image_n.width
    shape = (height, width)

    image_vis = []
    for x in range(1, len(CLASSES)+1):
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

    # Buka file GeoTIFF
    with rasterio.open(temp_file) as src:
        # Baca data piksel untuk band B4 dan B8
        red_band = src.read(4)  # Band 4 corresponds to B4 (Red)
        nir_band = src.read(8)  # Band 8 corresponds to B8 (NIR)

        # Hitung NDVI
        ndvi = (nir_band - red_band) / (nir_band + red_band)

        # Membatasi hasil NDVI ke rentang -1 hingga 1
        ndvi = ndvi.clip(-1, 1)
        ndvi = np.nan_to_num(ndvi, nan=1.0)

        # Klasifikasi NDVI
        classified_ndvi = classify_ndvi(ndvi)

    # Bentuk array data X (gambar) dan Y (label)
    X = np.stack((red_band.flatten(), nir_band.flatten()), axis=1)  # Stack bands B4 and B8 to create image array
    y = classified_ndvi.flatten()  # Flatten the classified NDVI array to create label array

    # Label encoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Convert samples dataframe (pandas) to numpy array
    test_input = reshape_input(np.array(X))

    # Also make label data to categorical
    test_output = to_categorical(y, N_CLASSES)

    n_model = load_model(model_path)

    pred     = n_model.predict(test_input, batch_size=1024)
    pred_img = np.argmax(pred, 1)
    pred_img = pred_img.reshape(shape[0], shape[1])

    tahun = Tahun()
    tahun.tahun = request.form['yearInput']
    tahun.save()

    # Menghitung jumlah piksel untuk setiap nilai unik dalam data
    unique_values, counts = np.unique(pred_img, return_counts=True)

    # Luas tiap piksel dalam meter persegi
    area_per_pixel_m2 = scale**2

    # Konversi luas dari meter persegi ke kilometer persegi (1 km² = 1,000,000 m²)
    area_per_pixel_km2 = area_per_pixel_m2 / 1_000_000

    # Menghitung luas dalam km² untuk setiap nilai
    area_km2 = counts * area_per_pixel_km2

    # Membulatkan hasil ke dua angka di belakang koma
    area_km2_rounded = np.round(area_km2, 2)

    # Membuat dictionary untuk menyimpan luas setiap nilai dalam km²
    area_dict_km2 = dict(zip(unique_values, area_km2_rounded))

    hasil = Hasil()
    hasil.tahun_id = tahun.serialize()['id']
    hasil.luas_air = area_dict_km2.get(0, 0)
    hasil.luas_lahan_terbangun = area_dict_km2.get(1, 0)
    hasil.luas_vegetasi = area_dict_km2.get(2, 0)
    hasil.save()

    flash('Proses training selesai & data berhasil disimpan.!', 'success')
    return redirect(url_for('data_index'))

def detail_data(id):
    tahun = Tahun.where('id', id).first().serialize()

    df_luas = Hasil.where('tahun_id', tahun['id']).get().serialize()

    temp_file = 'static/ndvi_image/'+ str(tahun['tahun']) +'.tif'

    # Load image
    image = rasterio.open(temp_file)
    # bandNum = image.count
    height = image.height
    width = image.width
    plot_size = (8, 8)
    shape = (height, width)

    # Buka file GeoTIFF
    with image as src:
        # Baca data piksel untuk band B4 dan B8
        red_band = src.read(4)  # Band 4 corresponds to B4 (Red)
        nir_band = src.read(8)  # Band 8 corresponds to B8 (NIR)

        # Hitung NDVI
        ndvi = (nir_band - red_band) / (nir_band + red_band)

        # Membatasi hasil NDVI ke rentang -1 hingga 1
        ndvi = ndvi.clip(-1, 1)
        ndvi = np.nan_to_num(ndvi, nan=1.0)

        # Klasifikasi NDVI
        classified_ndvi = classify_ndvi(ndvi)

    # Bentuk array data X (gambar) dan Y (label)
    X = np.stack((red_band.flatten(), nir_band.flatten()), axis=1)  # Stack bands B4 and B8 to create image array
    y = classified_ndvi.flatten()  # Flatten the classified NDVI array to create label array

    # Label encoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Convert samples dataframe (pandas) to numpy array
    test_input = reshape_input(np.array(X))

    # Also make label data to categorical
    test_output = to_categorical(y, N_CLASSES)

    n_model = load_model(model_path)

    # Predict test data
    prediction = np.argmax(n_model.predict(test_input, batch_size=1024), 1).flatten()
    label      = np.argmax(test_output, 1).flatten()

    cm           = confusion_matrix(label, prediction)
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

    pred     = n_model.predict(test_input, batch_size=1024)
    pred_img = np.argmax(pred, 1)
    pred_img = pred_img.reshape(shape[0], shape[1])

    # Buat colormap dan normalisasi warna
    cmap = mcolors.ListedColormap(PALETTE)
    norm = mcolors.BoundaryNorm(boundaries=[-1, 1, 1.5, 2.5], ncolors=3)

    # Plot bands dengan palet warna yang telah ditentukan
    ep.plot_bands(pred_img, cmap=cmap, norm=norm, figsize=plot_size)
    plt.savefig('static/ndvi_image/hasil_'+ str(tahun['tahun']) +'.jpg')
    plt.close()

    return render_template('pages/detail_data.html', accuracy_=accuracy_, precision_=precision_, recall_=recall_, 
    f1_score_=f1_score_, tahun=tahun['tahun'], df_luas=df_luas, segment='data')

# Function to reshape array input
def reshape_input(array):
    shape = array.shape
    return array.reshape(shape[0], shape[1], 1)

# Fungsi untuk mengklasifikasikan nilai NDVI
def classify_ndvi(ndvi_values):
    classified_ndvi = np.empty_like(ndvi_values, dtype='object')
    classified_ndvi[(ndvi_values <= 0)] = "Air"
    classified_ndvi[((ndvi_values > 0) & (ndvi_values <= 0.2))] = "Lahan Bangunan"
    classified_ndvi[(ndvi_values > 0.2)] = "Vegetasi"
    return classified_ndvi
