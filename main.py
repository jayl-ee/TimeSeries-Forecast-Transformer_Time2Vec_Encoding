from Transformer import *
from my_function import *

import argparse
import datetime
import yaml

# with open('/config/config.yaml') as f:
#     config = yaml.safe_load(f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--keep_train', type=str ,default='n')
    parsers = parser.parse_args()

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
scaler_p = Scaler(train=train, test=test)



if parsers.type == 'train':
    train_param = config['training_parameter']

    INPUT_COL = train_param['input_col']
    seq_len = train_param['seq_len']
    output_distance = train_param['output_distance']
    d_k = train_param['d_k']
    d_v = train_param['d_v']
    ff_dim = train_param['ff_dim']
    n_heads = train_param['n_heads']
    batch_size = train_param['batch_size']
    epoch = train_param['epoch']
    ver_size = train_param['ver_size']
    cmp_size = train_param['cmp_size']

    train['price'], test['price'] = scaler_p.MinMaxScale('price')

    train['version'], test['version'] = scaler_p.LabelEncode(train_param['label_dict'], 'version')
    train['company'], test['company'] = scaler_p.LabelEncode(train_param['label_dict'], 'company')



    in_out = Prepare_InputOutput(train, test,input_col=INPUT_COL, output_col='price',cat_col=['version','company'],train_ratio=0.8)

    train_x_p, val_x_p, train_x_cat1,train_x_cat2,val_x_cat1,val_x_cat2, train_y, val_y = in_out.Prepare_TrainVal( window_size=seq_len,output_distance=output_distance,)
    test_x_p,test_x_cat1,test_x_cat2, test_y = in_out.Prepare_Test(window_size=seq_len, output_distance=output_distance,version=1, company=1)

    train_x_1 = np.asarray(train_x_p).reshape((len(train_x_p),seq_len,len(INPUT_COL)))
    train_x_2 = np.asarray(train_x_cat1).reshape((len(train_x_p),seq_len,1))
    train_x_3 = np.asarray(train_x_cat2).reshape((len(train_x_p),seq_len,1))
    train_y = np.asarray(train_y).reshape((len(train_x_p),1))

    val_x_1 = np.asarray(val_x_p).reshape((len(val_x_p),seq_len,len(INPUT_COL)))
    val_x_2 = np.asarray(val_x_cat1).reshape((len(val_x_p),seq_len,1))
    val_x_3 = np.asarray(val_x_cat2).reshape((len(val_x_p),seq_len,1))
    val_y = np.asarray(val_y).reshape((len(val_x_p),1))

    test_x_1 = np.asarray(test_x_p).reshape((len(test_x_p),seq_len,len(INPUT_COL)))
    test_x_2 = np.asarray(test_x_cat1).reshape((len(test_x_p),seq_len,1))
    test_x_3 = np.asarray(test_x_cat2).reshape((len(test_x_p),seq_len,1))
    test_y = np.asarray(test_y).reshape((len(test_x_p),1))

    

    callback = tf.keras.callbacks.ModelCheckpoint(f'ckpt/Transformer+TimeEmbedding+LabelEmbedding_lead{output_distance}_window{seq_len}.hdf5', 
                                              monitor='val_loss', 
                                              save_best_only=True, verbose=1)

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1,patience=3)

    if parsers.keep_train == 'y':

        model = tf.keras.models.load_model(f'ckpt/Transformer+TimeEmbedding+LabelEmbedding_lead{output_distance}_window{seq_len}.hdf5',
                                custom_objects={'Time2Vector': Time2Vector, 
                                                'SingleAttention': SingleAttention,
                                                'MultiAttention': MultiAttention,
                                                'TransformerEncoder': TransformerEncoder,
                                                'RMSE':RMSE})
    else:
        model = create_model(seq_len, d_k, d_v, n_heads, ff_dim, output_distance, ver_size, cmp_size, INPUT_COL)

    history = model.fit([train_x_1,train_x_2,train_x_3], train_y, 
                    batch_size=batch_size, 
                    epochs=epoch, 
                    callbacks=[callback,earlystop],
                    validation_data=([val_x_1,val_x_2,val_x_3], val_y))  



if parsers.type == 'test':

    test_param = config['testing_parameter']
    INPUT_COL = test_param['input_col']
    seq_len = test_param['seq_len']
    output_distance = test_param['output_distance']
    d_k = test_param['d_k']
    d_v = test_param['d_v']
    ff_dim = test_param['ff_dim']
    n_heads = test_param['n_heads']
    batch_size = test_param['batch_size']
    epoch = test_param['epoch']
    ver_size = test_param['ver_size']
    cmp_size = test_param['cmp_size']
    

    inference = pd.read_csv('data/inference.csv')

    _, inference['price'] = scaler_p.MinMaxScale('price')
    _, inference['version'] = scaler_p.LabelEncode(test_param['label_dict'], 'version')
    _, inference['company'] = scaler_p.LabelEncode(test_param['label_dict'], 'company')

    # inference['price'] = inference['price'].diff(1)

    in_out = Prepare_InputOutput(None, inference,input_col=INPUT_COL, output_col='price',cat_col=['version','company'],train_ratio=0.8)
    inference_x_p,inference_x_cat1,inference_x_cat2, inference_y = in_out.Prepare_Test(window_size=seq_len, output_distance=output_distance,version=1, company=1)

    inference_x_1 = np.asarray(inference_x_p).reshape((len(inference_x_p),seq_len,len(INPUT_COL)))
    inference_x_2 = np.asarray(inference_x_cat1).reshape((len(inference_x_p),seq_len,1))
    inference_x_3 = np.asarray(inference_x_cat2).reshape((len(inference_x_p),seq_len,1))
    inference_y = np.asarray(inference_y).reshape((len(inference_x_p),1))

    model = tf.keras.models.load_model(f'ckpt/Transformer+TimeEmbedding+LabelEmbedding_lead{output_distance}_window{seq_len}_volume.hdf5',
                                   custom_objects={'Time2Vector': Time2Vector, 
                                                   'SingleAttention': SingleAttention,
                                                   'MultiAttention': MultiAttention,
                                                   'TransformerEncoder': TransformerEncoder,
                                                  'RMSE':RMSE})

    pred = model.predict([inference_x_1, inference_x_2, inference_x_3])

    df = pd.DataFrame({'actual':inference_y[:-seq_len].reshape(-1), 'predict':pred[seq_len:].reshape(-1)})
    
    now = datetime.datetime.now().strftime('%Y-%m-%d')

    df.to_csv(f'result_{now}.csv')