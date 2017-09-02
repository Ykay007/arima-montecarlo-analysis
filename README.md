# python-arima-montecarlo

Summary: The purpose of this repository is to illustrate how to conduct ARIMA Modelling and Monte Carlo Simulations on a simulated dataset using Python.

1. Firstly, the dataset is generated in database format using PostgreSQL.

2. The <strong>sqlalchemy</strong> library is used to import the SQL data into the Python environment.

3. Autocorrelation and partial autocorrelation plots are generated, and an <strong>ARIMA (0,1,0)</strong> model is used to forecast the last 100 observations in the time series.

4. The ARIMA predictions showed an overall mean deviation of X% to the actual observations.
