# TimeSeries-Forecast-Transformer_Time2Vec_Encoding
Time Series Forecast Model using Transformer &amp; Time2Vec &amp; Label Encoding on categorical data

This Model predicts time series data using numerical data & categorical data.
For instance, if you are to predict price of a laptop, below would be the input variables.

numerical data : price
categorical data : company, version



### 📁code
* #### my_function.py | contains 3 Classes.
- Scaler : includes MinMaxScaling, Reverse MinMaxScaling, Label Encoding
- Prepare_InputOutput : converts data into window sized sequence data, whilst splitting into Train/Val/Test dataset
- Preprocess : provides averageing values for which data is duplicated within same date
- PricePlot : Plots value with Plotly Library.

* #### Preprocess.py | file to run before running main.py
- Data cleaning code before training the model.
- If you have multiple cateforical data, please modify the code.

* #### Transformer.py
- Vanilla Scaled-dot-prodcuct Attention based Transformer
- Time2Vec
- LabelEncoder (scikit-learn)

### 📁data
- ** Please read config.yaml & my_function.py when you are using your own data with different COLUMN NAMES

* #### Config.yaml

```yaml
## Example
training_parameter:
  input_col : ['price']
  epoch : 35
  ver_size : 6
  cmp_size : 5
  batch_size : 32

  seq_len : 21

  output_distance : 5

  d_k : 128
  d_v : 128

  n_heads : 3
  ff_dim : 16

  label_dict : {'version':['1','2','3','4','5'] , 'company' : ['Apple','Samsung','Xiaomi','Lenovo'] }
```

```python
class Prepare_InputOutput:
    def __init__ (self, train_df, test_df, input_col, output_col, cat_col, train_ratio):
        -
        -
        -
    def Prepare_TrainVal( self, window_size , output_distance, ): # input_col : version , price, company
        
        ver_unique = self.train_df['version'].unique() 
        cmp_unique = self.train_df['company'].unique()


        input_price_data = []
        input_cat1_data = []
        input_cat2_data = []
        output_data = []
```