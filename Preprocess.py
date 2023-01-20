from my_function import *

########################

test_ratio = 0.3 

########################

df = pd.read_csv('data/data.csv')

preprocess = Preprocess(df)

df_tmp = preprocess.Per_version_company_AverageDuplicated()
df_tmp.dropna(subset=['price'], inplace=True)
df_tmp['date'] = pd.to_datetime(df_tmp['date'], format='%Y-%m-%d')
df_tmp = df_tmp.sort_values(['version','company','date',]) 
df_tmp.reset_index(drop=True, inplace=True)

train = df_tmp.copy()
test = pd.DataFrame(columns = df_tmp.columns)

ver_unique = df_tmp['version'].unique()
cmp_unique = df_tmp['company'].unique()




for ver in tqdm(ver_unique):
    for cmp in cmp_unique:

      tmp = train[np.logical_and(train['version']==ver, train['company']==cmp)].copy()
      tmp.sort_values('date',ascending=False,  inplace=True) #top 0.3 values will be the test values; ascending = False

      test_num = round(len(tmp) * test_ratio)
      test_idx = tmp.iloc[:test_num,:].index

      test = test.append(train.loc[test_idx,:])
      train.drop(index=test_idx, inplace=True)
      

test.to_csv('data/test.csv')
train.to_csv('data/train.csv')

