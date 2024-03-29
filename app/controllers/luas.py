from flask import render_template
from app.models.Tahun import *
from app.models.Hasil import *
import pandas as pd

def index():
    data = Tahun.join('hasil', 'hasil.tahun_id', '=', 'tahun.id').select('hasil.*', 'tahun.tahun').get().serialize()
    df_hasil = pd.DataFrame(data)
    return render_template('pages/luas.html', df_hasil=df_hasil, segment='luas')