import csv
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

time = []
open_amount = []

def data_pull(filename):
	with open(filename,'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			time.append(int(row[0]))
			open_amount.append(float(row[1]))
	return

def price_model(time,open_amount,future_date):
	linearity = linear_model.LinearRegression()
	time = np.reshape(time,(len(time),1))
	open_amount = np.reshape(open_amount,(len(open_amount),1))
	linearity.fit(time,open_amount)
	future_price =linearity.predict(future_date)
	return future_price[0][0],linearity.coef_[0][0] ,linearity.intercept_[0]


def show_plot(time,open_amount):
	linearity = linear_model.LinearRegression()
	time = np.reshape(time,(len(time),1))
	open_amount = np.reshape(open_amount,(len(open_amount),1))
	linearity.fit(time,open_amount)
	plt.scatter(time,open_amount,color='red')
	plt.plot(time,linearity.predict(time),color='blue',linewidth=2) 
	plt.show()
	return

data_pull('facebook.csv')
print time
print open_amount

show_plot(time,open_amount)


future_price, coefficient, constant = price_model(time,open_amount,12)  
print "Thank you for using Naresh's Stock Price Predictor. Facebook's ($FB) estimated stock price tomorrow morning will be: $",str(future_price)
