# ML-timeseries-project

This project has been created by Sriayndass Adidass and Reet Barik as a part of the Machine Learning coursework. Here we try different timeseries prediction models on techonogy stock data from 2013 - 2018 to find out if adding NIFTY (equivalent to NASDAQ in India) as the exogenous feature to a stock betters the prection of the stock. This is based on the intuition that NIFTY acts as an ecapsulation of all the events (such as political news, petrol price, exchange rate, etc.) which very much affect the price of the stock. We have considered technology stocks only for now because techology industry has been driving economy for the past few years in India (expecially in th 5 year period we are considering) and if we prove the hypothesis is correct here, there is high chance that it is true for other stocks too.

To prove/disprove this we tried various timeseries prediction methods from SARIMAX to GRUs. Prerequisites for the code are: Numpy, Pandas, Sklearn, Matplotlib, Statsmodels, Tensorflow and Keras. The following are the observations for each one:
1. SARIMAX: As expected this model doesn't perform well at all but such pseudo-linear univariate models usually act as a form of analysis and it was just that. We learnt about the optimal seasonality of the data, which we then used in our deep learning models too. Also, this doesn't accept exogenous variables, so it does nothing to prove/disprove our hypothesis.
2. Holt's Winter: Also doesn't do much as it is also univariate and doesn't accept exogenous variables. 
3. VAR: It is a type of pseudo-linear model that accepts exogenous variables and is multivate. But we see that it doesn't perform very well. But being the first model we tried that accepts exogenous data, the results are not promising for NIFTY doesn't seem to offer anything to prediction of the stock data.
4. XGBoost: Being the gold standard for efficient models, XGBoost gives error that's very good and that we expect to see. So any results that we get from this is pretty sound and we see that, again, there is nothing that NIFTY can offer to make the stock prection better. So we are close to rejecting our null hypothesis but to drive this result out the park we need more accuate models.
5. LSTM: We see that LSTMs, as expected give, extremely good models with the minimal amount of tuning that we did. Again as expected now, we see that there is no substancial improvement in prediction accuracy when NIFTY is fed into the network. Sometimes models even perform worse than before.
6. GRU: The observations made in LSTMs are valid here also. This is consistant with the fact that GRUs have just evolved from LSTMs and offer no benifit other than a performance gain. As to which model perform better depends on the data .

With the observations that we have made from all the algorithms we can reject our hypothesis that NIFTY will improve the prediction accuracy of technology stock data.

TODO:
1. Try on other types of stocks not just technology.
2. Better tuning on stock data for the deep-learning models.
3. Try to find more exogenous variables like comodities values to see if all of them together improve acuracy.
4. Increase time-frame of data.