import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, LabelEncoder
plt.rcParams['font.family'] = 'AppleGothic'
# %matplotlib inline




class Scaler:

    def __init__(self, train, test ):
                
        self.train = train
        self.test = test
        columns = 'price'

        self.mm_scale = MinMaxScaler()
        self.mm_scale.fit(self.train[[columns]])


    def MinMaxScale(self,columns):

        scaled_train = self.mm_scale.transform(self.train[[columns]])
        scaled_test = self.mm_scale.transform(self.test[[columns]])

        return scaled_train, scaled_test

    def ReverseMinMax(self, target, columns):
        self.target = target
        reverseminmax = self.mm_scale.inverse_transform(self.target[[columns]])

        return reverseminmax
    
    def LabelEncode(self, dictionary, column ):
        self.dict = dictionary # dictionary in { 'COLUMN NAME' : 'UNIQUE_verIABLES' }


        self.encoder = LabelEncoder()
        self.encoder.fit(self.dict[column])
        train_encoded = self.encoder.transform(self.train[column])
        test_encoded = self.encoder.transform(self.test[column])

        train_encoded += 1
        test_encoded += 1

        return train_encoded, test_encoded



class Prepare_InputOutput:
    def __init__ (self, train_df, test_df, input_col, output_col, cat_col, train_ratio):

        self.train_df = train_df
        self.test_df = test_df
        self.input_col = input_col
        self.output_col = output_col
        self.cat_col = cat_col
        self.train_ratio = train_ratio

    def Prepare_TrainVal( self, window_size , output_distance, ): # input_col : version , price, company
        
        ver_unique = self.train_df['version'].unique()
        cmp_unique = self.train_df['company'].unique()


        input_price_data = []
        input_cat1_data = []
        input_cat2_data = []
        output_data = []



        for ver in tqdm(ver_unique):

            for cmp in cmp_unique:

                ws = window_size
                input_output_length = ws + output_distance
                output_loc = input_output_length - 1
                
                tmp = self.train_df[np.logical_and(self.train_df['version']==ver, self.train_df['company']==cmp)].copy()
                tmp.sort_values('date',ascending=True,  inplace=True) 

                df_input_pr = tmp[self.input_col].values
                df_input_cat1 = tmp[self.cat_col[0]].values
                df_input_cat2 = tmp[self.cat_col[1]].values
                df_output = tmp[self.output_col].values

                for step in range(len(tmp)- input_output_length + 1):
                    input_price_data.append(df_input_pr[step : ws])
                    input_cat1_data.append(df_input_cat1[step : ws])
                    input_cat2_data.append(df_input_cat2[step : ws])
                    output_data.append(df_output[output_loc])

                    ws += 1
                    output_loc += 1

        train_len = round( len(input_price_data) * self.train_ratio)

        train_x_p = input_price_data[:train_len]
        train_x_cat1 = input_cat1_data[:train_len]
        train_x_cat2 = input_cat2_data[:train_len]
        train_y = output_data[:train_len]


        val_x_p = input_price_data[train_len:]
        val_x_cat1 = input_cat1_data[train_len:]
        val_x_cat2 = input_cat2_data[train_len:]
        val_y = output_data[train_len:]

        return train_x_p, val_x_p, train_x_cat1,train_x_cat2,val_x_cat1,val_x_cat2, train_y, val_y


    def Prepare_Test(self,window_size, output_distance, version:int, company:int ):
        test_x_p = []
        test_x_cat1 = []
        test_x_cat2 = []
        test_y = []

      
        input_output_length = window_size + output_distance
   

        tmp = self.test_df[np.logical_and(self.test_df['version']==version, self.test_df['company']==company)].copy()
        tmp.sort_values('date',ascending=True,  inplace=True)

        df_input_pr = tmp[self.input_col].values
        df_input_cat1 = tmp[self.cat_col[0]].values
        df_input_cat2 = tmp[self.cat_col[1]].values
        df_output = tmp[self.output_col].values

        output_loc = input_output_length - 1

        for step in tqdm(range(len(tmp)- input_output_length + 1 )):
                    test_x_p.append(df_input_pr[step : window_size])
                    test_x_cat1.append(df_input_cat1[step : window_size])
                    test_x_cat2.append(df_input_cat2[step : window_size])
                    test_y.append(df_output[output_loc])
             

                    window_size += 1
                    output_loc += 1

        return test_x_p,test_x_cat1,test_x_cat2, test_y








class Preprocess:

    def __init__(self, df):
        self.df = df

    def nan_ck (self, colname):
        self.colname = colname
        return self.df[self.df[self.colname].isnull()]



    def Yearly_cnt( self, criteria, top ): #criteria :"market / region" 을 기준으로 연간으로 건수 시각화 --> barplot
        ''' '''

        self.criteria = criteria
        self.top = top

        df = self.df
        df.index = pd.to_datetime(df['date'], format='%Y-%m-%d')

        twenty = df.loc['2020-01-01':'2020-12-31',:]
        twentyone = df.loc['2021-01-01':'2021-12-31',:]
        twentytwo = df.loc['2022-01-01':'2022-12-31',:]

        lst = [twenty, twentyone, twentytwo]

        for i, year in enumerate(lst):
            piv = year.pivot_table(index=[criteria], values='cnt', aggfunc='sum')
            piv = piv.reset_index()
            piv = piv.sort_values('cnt',ascending=False)

            sns.barplot(data=piv[:top], x='cnt',y=criteria)
            plt.show()

        return twenty, twentyone, twentytwo


    def AverageDuplicated( self ):


        mk_unique = self.df['market'].unique() #market list
        

        for market in tqdm(mk_unique):
            tmp = self.df[self.df['market']==market].copy()

            if tmp['region'].isnull().sum() != 0:
                tmp['region'].fillna('-', inplace=True)

            
            rg_unique = tmp['region'].unique() #region list in certain market

            for region in rg_unique:
                tmp = tmp[tmp['region']==region].copy()
                # print(tmp2['market'].unique(), region)
                
                ### Handle Duplicates
                tmp = tmp[tmp.duplicated(subset=['date','company'], keep=False)].copy()


                date_unique = tmp['date'].unique() #date list
                company_unique = tmp['company'].unique() #company list

                for company in company_unique:
                    tmp = tmp[tmp['company']==company].copy()
                    tmp = tmp.sort_values('date', ascending=False)
            

                    for date in date_unique:
                        duplicate_idx = tmp[tmp['date']== date].index.values
                        # print(duplicate_idx)

                        ## Update average prices to the first row
                        try:
                            if len(duplicate_idx) > 0:
                                update_idx = duplicate_idx[0]
                                remove_idx = duplicate_idx[1:]

                                self.df.loc[update_idx, ['price','low_price','high_price']] = self.df.loc[duplicate_idx,['price','low_price','high_price']].mean(axis=0)
                                self.df = self.df.drop(index=remove_idx)
                                # df.reset_index(inplace=True, drop=False)  

                        except : 
                            continue 


        return self.df           

    def Per_version_company_AverageDuplicated( self ):



            va_unique = self.df ['version'].unique() #market list
            

            for ver in tqdm(va_unique):
                tmp = self.df [self.df ['version']==ver]


                tmp_1 = tmp[tmp.duplicated(subset=['date','company'], keep=False)]


                date_unique = tmp_1['date'].unique() #date list
                company_unique = tmp_1['company'].unique() #company list

                for company in company_unique:
                    tmp_2 = tmp_1[tmp_1['company']==company]
                    tmp_2 = tmp_2.sort_values('date', ascending=False)
            

                    for date in date_unique:
                        duplicate_idx = tmp_2[tmp_2['date']== date].index.values
                        # print(duplicate_idx)

                        ## Update average prices to the first row
                        try:
                            if len(duplicate_idx) > 0:
                                update_idx = duplicate_idx[0]
                                remove_idx = duplicate_idx[1:]

                                self.df .loc[update_idx, ['price']] = self.df .loc[duplicate_idx,['price']].mean(axis=0)
                                self.df  = self.df .drop(index=remove_idx)
                                # df.reset_index(inplace=True, drop=False)  

                        except : 
                            continue 


            return self.df       



class PricePlot:

    def __init__ (self, df):
        self.df = df

    def plot_price_rg( self, region ):
        self.region = region

        tmp = self.df.loc[self.df['region']==self.region]
        plt.figure(figsize=(40,10))
        df = pd.DataFrame({"price": tmp['price'],
                            'low_price': tmp['low_price'],
                            'high_price':tmp['high_price'],
                            'date':tmp['date']})
        df.index= df['date']
        df.sort_index(inplace=True)

        fig = px.line(df, x=df['date'], y=['price','low_price','high_price'])
        fig.show()


    def plot_price_mk( self, market ):
        self.market = market


        tmp = self.df.loc[self.df['market']== self.market]
        plt.figure(figsize=(40,10))
        
        df = pd.DataFrame({"price": tmp['price'],
                            'low_price': tmp['low_price'],
                            'high_price':tmp['high_price'],
                            'date':tmp['date']})
        df.index= df['date']
        df.sort_index(inplace=True)
        fig = px.line(df, x=df['date'], y=['price','low_price','high_price'])
        fig.show()


    def PlotPriceBy_mkrg(self, region ):


        self.region = region
        tmp = self.df.loc[self.df['region']==self.region] #tmp dataframe by region
        
        markets = tmp['market'].unique() #markets in certain region ; numpy array

        for i, mkt  in enumerate(markets):
        
            tmp_mk = tmp[tmp['market']==mkt]

            # plt.subplot(num_mk*100 + 10 + i + 1)
            plt.figure(figsize=(30,5))

            data = pd.DataFrame({"price": tmp_mk['price'],
                            'low_price': tmp_mk['low_price'],
                            'high_price':tmp_mk['high_price'],
                            'date':tmp_mk['date']})
            data.index= data['date']
            data.sort_index(inplace=True)

            fig = px.line(data, x=data['date'], y=['price','low_price','high_price'] )
            print(f'{self.region} / {mkt} ')
            fig.show()
    
