# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

# import the necessary packages
import pickle
import pandas as pd
import os
#import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
import time

from model_transforms import NumberTaker, ExperienceTransformer, NumpyToDataFrame

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
	# load the pre-trained model
	# global model
	with open(model_path, 'rb') as f:
		model = pickle.load(f)
	print(model)
	return model

modelpath = "/app/app/models/ctb_clf.pkl"
# modelpath = "C:/Users/Admin/Desktop/ddekun_docker_flask_example/training_files/models/ctb_clf.pkl"
load_model(modelpath)


@app.route("/", methods=["GET"])
def general():
	return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	response = {'success': False}
	curr_time = time.strftime('[%Y-%b-%d %H:%M:%S]')

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == 'POST':
		request_json = flask.request.get_json()

		input_data = pd.DataFrame({
			'enrollee_id': None,
			'city': request_json.get('city', ''),
			'city_development_index': float(request_json.get('city_development_index', '')),
			'gender': request_json.get('gender', ''),
			'relevent_experience': request_json.get('relevent_experience', ''),
			'enrolled_university': request_json.get('enrolled_university', ''),
			'education_level': request_json.get('education_level', ''),
			'major_discipline': request_json.get('major_discipline', ''),
			'experience': request_json.get('experience', ''),
			'company_size': request_json.get('company_size', ''),
			'company_type': request_json.get('company_type', ''),
			'last_new_job': request_json.get('last_new_job', ''),
			'training_hours': int(request_json.get('training_hours', '')),
		}, index=[0])
		logger.info(f'{curr_time} Data: {input_data.to_dict()}')

		try:
			# Predict the result
			preds = model.predict_proba(input_data)
			response['predictions'] = round(preds[:, 1][0], 5)
			# Request successful
			response['success'] = True
		except AttributeError as e:
			logger.warning(f'{curr_time} Exception: {str(e)}')
			response['predictions'] = str(e)
			# Request unsuccessful
			response['success'] = False

	# return the data dictionary as a JSON response
	return flask.jsonify(response)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 8180))
	app.run(host='0.0.0.0', debug=True, port=port)
