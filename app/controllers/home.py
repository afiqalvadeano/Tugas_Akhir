from flask import render_template, flash, redirect, url_for, request
import ee

def index():
    return render_template('pages/home.html', segment='home')