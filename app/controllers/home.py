from flask import render_template
import geemap
import ee
def index():
    ee.Authenticate()
    ee.Initialize(project='1031392701041')
    # Define the region of interest (ROI)
    roi = ee.Geometry.Rectangle([110.0089, -8.2414, 110.8603, -7.4959])

    # Define the time range
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    # Create an image collection for Sentinel-2
    sentinel2 = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(roi) \
        .filterDate(ee.Date(start_date), ee.Date(end_date)) \
        .median()

    # Calculate NDVI
    ndvi = sentinel2.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # Clip the NDVI image to the specified ROI
    ndvi_roi = ndvi.clip(roi)

    # Convert the NDVI image to a visualization-friendly format
    ndvi_vis = ndvi_roi.visualize(min=-1, max=1, palette=['blue', 'white', 'green'])

    # Save the NDVI image to a temporary file
    temp_file = 'ndvi_image.tif'
    geemap.ee_export_image(ndvi_vis, filename=temp_file, scale=1000, region=roi, file_per_band=False)
    print('cek')

    return render_template('pages/home.html', segment='home')