#!flask/bin/python

import sys
from flask import Flask, render_template, request, redirect, Response
import random
import json


# setup the flask server
app = Flask(__name__)

# the route to be the main index page


@app.route("/")
def main_page():
    return render_template('index.html')


# receive input from javascript
@app.route('/postmethod', methods = ['POST'])
def get_post_javascript_data():
    jsdata = request.form['javascript_data']
    print("The jsdata is", jsdata)
    return jsdata

    
@app.route('/getmethod')
def get_python_data():
	pythondata = "This is the answer hahahaha"
	return json.dumps(pythondata)


if __name__ == "__main__":
    app.run()
