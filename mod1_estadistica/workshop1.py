import numpy as np
import pandas as pd
import pandas_datareader as pdr
from matplotlib import pyplot

#Getting BTC information since 01/01/2017
BTC = pdr.get_data_yahoo('BTC-USD', start = "01/01/2017", interval="d")
#Getting BTC returns
BTC["R"] = (BTC["Close"] / BTC["Close"].shift(1)) - 1
print("The # of returns is", BTC["R"].count())
r_bitcoin = pd.DataFrame(BTC[["R"]])

#Setting x-axis and y-axis from plot, considering mean, standard deviation and number of returns
ccret = r_bitcoin.to_numpy()
x = np.random.normal(loc = BTC["R"].mean(), scale = BTC["R"].std(), size = BTC["R"].count())
y = ccret
bins = 12

#Histogram plot for simulated and real returns
pyplot.hist(x, bins, alpha = 0.5, label = 'simulated')
pyplot.hist(y, bins, alpha = 0.5, label = 'real')
pyplot.legend(loc='upper left')
pyplot.title(label='Histogram of real and simulated cc returns of Bitcoin')
pyplot.show()