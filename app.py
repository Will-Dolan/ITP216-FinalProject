'''
Will Dolan
Fall 2023
Tuesdays 6pm
Final Project
'''


import io
import os

import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for, send_file
from matplotlib.figure import Figure
from joblib import load
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.route("/")
def home():
	# set up dicts for selectors in templates
	regions = {
		'southwest': 0,
		'southeast': 1,
		'northwest': 2,
		'northeast': 3
	}
	ranges = {
		'<12754.2': 0, 
		'<25508.4': 1, 
		'<38262.6': 2, 
		'<51016.8': 3, 
		'<63771.0': 4
	}
	if not 'message' in session:
		session['message'] = None
	return render_template("home.html", message=session['message'], regions=regions, ranges=ranges)


@app.route("/request_old_data", methods=["POST"])
def request_old_data():
	'''
	return a figure of an old data point based on its label
	'''
	session["request"] = request.form["request"]
	return redirect(url_for("graph", data_request=session["request"], predict=False))


@app.route("/predict_data", methods=["POST"])
def predict_data():
	'''
	return a prediction of some user inputted data
	'''
	session["data"] = [
		request.form['age'],
		request.form['sex'],
		request.form['bmi'],
		request.form['smoker'],
		request.form['children'],
		request.form['region'],
	]
	try:
		session['data'] = list(map(int, session['data']))
		for i in session['data']:
			print(type(i))
	except:
		session['message'] = 'Make sure to input data correctly'
		return redirect(url_for('home'))
	
	return redirect(url_for("graph", data_request='predict', predict=True))


@app.route("/api/graph/<data_request>/<predict>", methods=["GET"])
def graph(data_request, predict):
	# set up dicts for selectors in templates
	regions = {
		'southwest': 0,
		'southeast': 1,
		'northwest': 2,
		'northeast': 3
	}
	ranges = {
		'<12754.2': 0, 
		'<25508.4': 1, 
		'<38262.6': 2, 
		'<51016.8': 3, 
		'<63771.0': 4
	}
	return render_template("graph.html", regions=regions, ranges=ranges, data_request=data_request, predict=predict)


@app.route("/fig/<data_request>/<predict>", methods=["GET"])
def fig(data_request, predict):
	# create a figure and send it to the template
	fig = create_figure(data_request, predict)

	img = io.BytesIO()
	fig.savefig(img, format='png')
	img.seek(0)
	return send_file(img, mimetype="image/png")


def create_figure(data_request, predict):
	# manipulation of the data done in separate file. 
	df = pd.read_csv('data.csv')
	
	if predict=='True':
		# load in saved model
		model = load('model.joblib')

		fig = Figure()
		ax = fig.add_subplot(1, 1, 1)
		fig.suptitle('Predicted range of insurance for the information listed below')
		data = session['data']
		info = f'Age: {data[0]}\n' \
				f'Sex: {"Male" if data[1] else "Female"} \n' \
				f'BMI: {data[2]} \n' \
				f'Smoker: {"Yes" if data[3] else "No"} \n' \
				f'Number of Children: {data[4]} \n' 
		data = pd.DataFrame(data).T

		# using the model, predict probabilities of each class
		y_hat = model.predict_proba(data)
		print(y_hat)

		# plot probabilities
		ax.bar(
			['<12754.2', '<25508.4', '<38262.6', '<51016.8', '<63771.0'], 
			[x for x in y_hat[0]]
		)
		ax.set(xlabel=f"Insurance Rate\n{info}", ylabel="Probability", title='Projected Insurance Rate Bin From Provided Information')
		ax.set_xticklabels(['<12754.2', '<25508.4', '<38262.6', '<51016.8', '<63771.0'])
		fig.set_tight_layout(True)

		return fig
	
	else:
		df = df[df.charges == float(data_request)]

		# get a random data point for the defined label
		idx = np.random.choice(len(df))
		data = df.iloc[idx]

		# make data readable
		if data['sex'] == 0:
			data['sex'] = 'female'
		else: data['sex'] = 'male'
		if data['smoker'] == 0:
			data['smoker'] = 'no'
		else: data['smoker'] = 'yes'

		regions = {
			0:'southwest',
			1:'southeast',
			2:'northwest',
			3:'northeast'
		}
		data['region'] = regions[int(data['region'])]

		# plot data
		fig = Figure()
		ax = fig.add_subplot(1, 1, 1)
		ax.bar(
			['<12754.2', '<25508.4', '<38262.6', '<51016.8', '<63771.0'], 
			[int(data_request) == x for x in range(5)]
		)
		ax.set(xlabel=f"Insurance Rate\n{data[1:-1].to_string()}", ylabel="Probability", title='Projected Insurance Rate Bin From Provided Information')
		ax.set_xticklabels(['<12754.2', '<25508.4', '<38262.6', '<51016.8', '<63771.0'])
		fig.set_tight_layout(True)
		return fig


@app.route('/<path:path>')
def catch_all(path):
	session['message'] ='Page not found'
	return redirect(url_for("home"))


if __name__ == "__main__":
	app.secret_key = os.urandom(12)
	app.run(debug=True)
