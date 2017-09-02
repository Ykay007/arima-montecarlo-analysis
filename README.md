# python-arima-montecarlo

Summary: The purpose of this repository is to illustrate how to conduct ARIMA Modelling and Monte Carlo Simulations on a simulated dataset using Python.

1. Firstly, the dataset is generated in database format using PostgreSQL.

2. The <strong>sqlalchemy</strong> library is used to import the SQL data into the Python environment.

3. Autocorrelation and partial autocorrelation plots are generated, and an <strong>ARIMA (0,1,0)</strong> model is used to forecast the last 100 observations in the time series.

4. The ARIMA predictions showed an overall absolute mean deviation of <strong>0.039%</strong> to the actual observations.

5. Then, a Monte Carlo simulation is generated to illustrate the range of potential deviations from the mean and their probabilities, with the distribution being calculated using the mean and standard deviation of the original time series.
