from flask import Flask, url_for, render_template, request
from flask import request


app = Flask(__name__)

@app.route('/')
def time():
   return render_template('index.html')


@app.route('/index',methods = ['POST', 'GET'])
def index():
   return render_template('index.html')
