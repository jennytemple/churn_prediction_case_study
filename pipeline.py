import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Pipeline(object):

	def __init__(self, model, X, target, parameters = {}, scaler = StandardScaler):
		'''
		Initializes the pipeline object
		INPUTS: 
			self
			model: an sklearn model class with fit and predict methods
			X: design matrix
			target: the value to be trained up
		OUTPUTS: 
			None
		'''
		self.model = model
		self.X1, self.X2, self.y1, self.y2 = train_test_split(X, target)
		self.parameters = parameters
 
		### Preparation attributes suite
		self.scaler = scaler()
		self.scaled_X1 = None
		self.scaled_y1 = None

		### Grid Search and Fit attributes suite
		self.final_X1 = None
		self.final_y1 = None
		self.fit_model = None
		self.prepare_pipe()

	def prepare_pipe(self):
		''' Prepare the pipe by fitting elements to the traning data '''
		self.not_prepared = True
		self.scaler.fit(self.X1)
		self.scaled_X1, self.scaled_y1 = self.transform(self.X1, self.y1)
		### Replace if everybody feels comfortable with the undersampler
		self.final_X1 = self.scaled_X1
		self.final_y1 = self.scaled_y1

	def transform(self, X, y = None):
		'''
		Transforms the data 
		INPUTS: 
			self
			X data
			y labels
		OUTPUTS: 
			Transformed X data
			Transformed y labels
		'''
		scaled_X1 = self.scaler.transform(X)
		scaled_y1 = y #TODO: transform y, maybe?
		return scaled_X1, scaled_y1

	def grid_search(self, cv = 3):
		'''
		INPUTS:
		    self
            cv = number of k=folds to use in cross validation
		OUTPUTS:
		        Sklearn Model with best parameters
		'''
		#if isnumber(self.final_X1):
		#	self.prepare_pipe()
		clf = GridSearchCV(self.model(), self.parameters, cv = cv)
		clf.fit(self.final_X1, self.final_y1.ravel())
		self.fit_model = clf.best_estimator_

	def fit(self, hyperparameters = {}):
		'''
		INPUTS: 
			self
			hyperparameters: a dictionary where each key is the name of the hyperparameter and 
				each value is the value with which we will fit
		OUTPUTS: 	
			model: sklearn model object
		'''
		'''
		if isinstance(self.final_X1, type(None)):
			self.prepare_pipe()
		'''
		if isinstance(self.fit_model, type(None)):
			self.fit_model = self.model(**hyperparameters)

		self.fit_model.fit(self.final_X1, self.final_y1)
		return self.fit_model

	def score(self, X = None, y = None):
		'''
		Score the pipeline model with testing data.
		INPUTS:
		        self
			X: Testing data (default self.X2)
			y: Testing labels (default self.y2)
		OUTPUT:
		        model score: float 
		'''
		if isinstance(X, type(None)):
			X, y = self.X2, self.y2
		scaled_X, scaled_y = self.transform(X, y)
		return self.fit_model.score(scaled_X, scaled_y)

	def predict(self, X):
		'''
		INPUTS:
		        self
			X: Data to predict from
		OUTPUTS:
		        y: predicted labels
		'''
		scaled_X, _ = self.transform(X)
		return self.fit_model.predict(scaled_X, scaled_y)
