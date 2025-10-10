from flask import Flask, request, jsonify, render_template
from flask_cors import cross_origin


app = Flask(__name__)

@app.route("/")
@cross_origin
def home_page():
    return render_template('index.html')

