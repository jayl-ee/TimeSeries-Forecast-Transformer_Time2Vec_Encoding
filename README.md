# TimeSeries-Forecast-Transformer_Time2Vec_Encoding
Time Series Forecast Model using Transformer &amp; Time2Vec &amp; Label Encoding on categorical data

This Model predicts time series data using numerical data & categorical data.
For instance, if you are to predict price of a laptop, below would be the input variables.

numerical data : price
categorical data : company, version




my_function.py | contains 3 Classes.
- Scaler : includes MinMaxScaling, Reverse MinMaxScaling, Label Encoding
- Prepare_InputOutput : converts data into window sized sequence data, whilst splitting into Train/Val/Test dataset
- Preprocess : provides averageing values for which data is duplicated within same date
- PricePlot : Plots value with Plotly Library.

Preprocess.py | file to run before running main.py
- Data cleaning code before training the model.


