import os
import gc
import datetime
import time
import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Reshape, Flatten
from keras.layers import concatenate, dot, add, multiply
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU, PReLU, ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop, Adam, SGD

from nn_generator import DataGenerator

# Set TEST_MODE = True for quick testing (fewer epochs, smaller data)
TEST_MODE = False  # Change to True for quick testing

def train_nn_model(random_seed, folder='training', config_index=0, test_mode=False):
    """
    Train neural network model with a specific random seed for train/val split.
    Returns predictions, ground truth, and metrics.
    """
    print(f'\n{"="*60}')
    print(f'Training with random seed: {random_seed}, config: {config_index}')
    print(f'{"="*60}\n')
    
    data_prep_start_time = time.time()
    
    ## load train data
    if test_mode:
        print("TEST_MODE: Loading only 10,000 rows for quick testing")
        nrows = 10000
    else:
        nrows = None
    
    if folder == 'training':
        if nrows:
            train_full = pd.read_csv('./input/%s/train_part.csv'%folder, nrows=nrows)
            train_add_full = pd.read_csv('./input/%s/train_part_add.csv'%folder, nrows=nrows)
        else:
            train_full = pd.read_csv('./input/%s/train_part.csv'%folder)
            train_add_full = pd.read_csv('./input/%s/train_part_add.csv'%folder)
    elif folder == 'validation':
        if nrows:
            train_full = pd.read_csv('./input/%s/train.csv'%folder, nrows=nrows)
            train_add_full = pd.read_csv('./input/%s/train_add.csv'%folder, nrows=nrows)
        else:
            train_full = pd.read_csv('./input/%s/train.csv'%folder)
            train_add_full = pd.read_csv('./input/%s/train_add.csv'%folder)

    # Extract target before splitting
    train_y_full = train_full['target'].copy()

    # Split data 80-20
    print(f'Splitting data with random seed: {random_seed}')
    train_indices, test_indices, train_y, test_y = train_test_split(
        train_full.index, train_y_full, 
        test_size=0.2, 
        random_state=random_seed,
        stratify=train_y_full  # maintain class distribution
    )

    # Split train and train_add using the same indices
    train = train_full.iloc[train_indices].reset_index(drop=True)
    test = train_full.iloc[test_indices].reset_index(drop=True)
    train_add = train_add_full.iloc[train_indices].reset_index(drop=True)
    test_add = train_add_full.iloc[test_indices].reset_index(drop=True)

    # Reset indices for targets
    train_y = train_y.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)

    # Drop target from both train and test
    train.drop(['target'], inplace=True, axis=1)
    test.drop(['target'], inplace=True, axis=1)

    print(f'Training set size: {len(train):,}')
    print(f'Validation set size: {len(test):,}')

    # Create test_id for saving (using index)
    test_id = test.index.values

    for col in train_add.columns:
        train[col] = train_add[col].values
        test[col] = test_add[col].values

    print('Train data loaded.')

    ## load other data
    member = pd.read_csv('./input/%s/members_nn.csv'%folder).sort_values('msno')
    song = pd.read_csv('./input/%s/songs_nn.csv'%folder).sort_values('song_id')

    member_add = pd.read_csv('./input/%s/members_add.csv'%folder).sort_values('msno')
    member_add.fillna(0, inplace=True)
    member = member.merge(member_add, on='msno', how='left')

    print('Member/Song data loaded.')

    ######################################################
    ## Feature Preparation
    ######################################################

    ## data preparation
    train = train.merge(member[['msno', 'city', 'gender', 'registered_via', \
            'registration_init_time', 'expiration_date']], on='msno', how='left')
    test = test.merge(member[['msno', 'city', 'gender', 'registered_via', \
            'registration_init_time', 'expiration_date']], on='msno', how='left')

    train = train.merge(song[['song_id', 'artist_name', 'composer', 'lyricist', \
            'language', 'first_genre_id', 'second_genre_id', 'third_genre_id', \
            'cc', 'xxx']], on='song_id', how='left')
    test = test.merge(song[['song_id', 'artist_name', 'composer', 'lyricist', \
            'language', 'first_genre_id', 'second_genre_id', 'third_genre_id', \
            'cc', 'xxx']], on='song_id', how='left')

    cols = ['song_id', 'artist_name', 'language', 'first_genre_id', 'song_rec_cnt']
    tmp = song[cols]

    tmp.columns = ['before_'+i for i in cols]
    train = train.merge(tmp, on='before_song_id', how='left')
    test = test.merge(tmp, on='before_song_id', how='left')

    tmp.columns = ['after_'+i for i in cols]
    train = train.merge(tmp, on='after_song_id', how='left')
    test = test.merge(tmp, on='after_song_id', how='left')

    train['before_type_same'] = (train['source_type'] == train['before_source_type']).astype(int)
    test['before_type_same'] = (test['source_type'] == test['before_source_type']).astype(int)

    train['after_type_same'] = (train['source_type'] == train['after_source_type']).astype(int)
    test['after_type_same'] = (test['source_type'] == test['after_source_type']).astype(int)

    train['before_artist_same'] = (train['artist_name'] == train['before_artist_name']).astype(int)
    test['before_artist_same'] = (test['artist_name'] == test['before_artist_name']).astype(int)

    train['after_artist_same'] = (train['artist_name'] == train['after_artist_name']).astype(int)
    test['after_artist_same'] = (test['artist_name'] == test['after_artist_name']).astype(int)

    train['before_genre_same'] = (train['first_genre_id'] == train['before_first_genre_id']).astype(int)
    test['before_genre_same'] = (test['first_genre_id'] == test['before_first_genre_id']).astype(int)

    train['after_genre_same'] = (train['first_genre_id'] == train['after_first_genre_id']).astype(int)
    test['after_genre_same'] = (test['first_genre_id'] == test['after_first_genre_id']).astype(int)

    del tmp
    gc.collect()

    print('Data preparation done.')

    ## generate data for training
    embedding_features = ['msno', 'city', 'gender', 'registered_via', \
            'artist_name', 'language', 'cc', \
            'source_type', 'source_screen_name', 'source_system_tab', \
            'before_source_type', 'after_source_type', 'before_source_screen_name', \
            'after_source_screen_name', 'before_language', 'after_language', \
            'song_id', 'before_song_id', 'after_song_id']

    train_embeddings = []
    test_embeddings = []
    for feat in embedding_features:
        train_embeddings.append(train[feat].values)
        test_embeddings.append(test[feat].values)

    genre_features = ['first_genre_id', 'second_genre_id']

    train_genre = []
    test_genre = []
    for feat in genre_features:
        train_genre.append(train[feat].values)
        test_genre.append(test[feat].values)

    context_features = ['after_artist_same', 'after_song_rec_cnt', 'after_timestamp', \
            'after_type_same', 'before_artist_same', 'before_song_rec_cnt', \
            'before_timestamp', 'before_type_same', 'msno_10000_after_cnt', \
            'msno_10000_before_cnt', 'msno_10_after_cnt', 'msno_10_before_cnt', \
            'msno_25_after_cnt', 'msno_25_before_cnt', 'msno_50000_after_cnt', \
            'msno_50000_before_cnt', 'msno_5000_after_cnt', 'msno_5000_before_cnt', \
            'msno_500_after_cnt', 'msno_500_before_cnt', 'msno_source_screen_name_prob', \
            'msno_source_system_tab_prob', 'msno_source_type_prob', 'msno_till_now_cnt', \
            'registration_init_time', 'song_50000_after_cnt', 'song_50000_before_cnt', \
            'song_till_now_cnt', 'timestamp', 'msno_artist_name_prob', 'msno_first_genre_id_prob', \
            'msno_xxx_prob', 'msno_language_prob', 'msno_yy_prob', 'msno_source_prob', \
            'song_source_system_tab_prob', 'song_source_screen_name_prob', 'song_source_type_prob']

    train_context = train[context_features].values
    test_context = test[context_features].values
    
    ss_context = StandardScaler()
    train_context = ss_context.fit_transform(train_context)
    test_context = ss_context.transform(test_context)
    
    del train
    del test
    gc.collect()

    usr_features = ['bd', 'expiration_date', 'msno_rec_cnt', 'msno_source_screen_name_0', \
            'msno_source_screen_name_1', 'msno_source_screen_name_10', 'msno_source_screen_name_11', \
            'msno_source_screen_name_12', 'msno_source_screen_name_13', 'msno_source_screen_name_14', \
            'msno_source_screen_name_17', \
            'msno_source_screen_name_18', 'msno_source_screen_name_19', 'msno_source_screen_name_2', \
            'msno_source_screen_name_20', 'msno_source_screen_name_21', 'msno_source_screen_name_3', 'msno_source_screen_name_4', \
            'msno_source_screen_name_5', 'msno_source_screen_name_6', 'msno_source_screen_name_7', \
            'msno_source_screen_name_8', 'msno_source_screen_name_9', 'msno_source_system_tab_0', \
            'msno_source_system_tab_1', 'msno_source_system_tab_2', 'msno_source_system_tab_3', \
            'msno_source_system_tab_4', 'msno_source_system_tab_5', 'msno_source_system_tab_6', \
            'msno_source_system_tab_7', 'msno_source_system_tab_8', \
            'msno_source_type_0', 'msno_source_type_1', 'msno_source_type_10', \
            'msno_source_type_11', 'msno_source_type_2', \
            'msno_source_type_3', 'msno_source_type_4', 'msno_source_type_5', \
            'msno_source_type_6', \
            'msno_source_type_7', 'msno_source_type_8', 'msno_source_type_9', \
            'msno_timestamp_mean', 'msno_timestamp_std', 'registration_init_time', \
            'msno_song_length_mean', 'msno_artist_song_cnt_mean', 'msno_artist_rec_cnt_mean', \
            'msno_song_rec_cnt_mean', 'msno_yy_mean', 'msno_song_length_std', \
            'msno_artist_song_cnt_std', 'msno_artist_rec_cnt_std', 'msno_song_rec_cnt_std', \
            'msno_yy_std', 'artist_msno_cnt']

    usr_feat = member[usr_features].values
    usr_feat = StandardScaler().fit_transform(usr_feat)

    song_features = ['artist_rec_cnt', 'artist_song_cnt', 'composer_song_cnt', \
            'genre_rec_cnt', 'genre_song_cnt', 'song_length', \
            'song_rec_cnt', 'song_timestamp_mean', 'song_timestamp_std', \
            'xxx_rec_cnt', 'xxx_song_cnt', 'yy', 'yy_song_cnt']

    song_feat = song[song_features].values
    song_feat = StandardScaler().fit_transform(song_feat)

    n_factors = 48

    usr_component_features = ['member_component_%d'%i for i in range(n_factors)]
    song_component_features = ['song_component_%d'%i for i in range(n_factors)]

    usr_component = member[usr_component_features].values
    song_component = song[song_component_features].values

    n_artist = 16

    usr_artist_features = ['member_artist_component_%d'%i for i in range(n_artist)]
    song_artist_features = ['artist_component_%d'%i for i in range(n_artist)]

    usr_artist_component = member[usr_artist_features].values
    song_artist_component = song[song_artist_features].values

    del member
    del song
    gc.collect()

    data_prep_end_time = time.time()
    data_prep_runtime = data_prep_end_time - data_prep_start_time
    print(f'\nData preparation runtime: {data_prep_runtime:.2f} seconds ({data_prep_runtime/60:.2f} minutes)')

    # Initialize data generator
    print('Initializing data generator...')
    dataGenerator = DataGenerator()
    train_flow = dataGenerator.flow(train_embeddings+train_genre, [train_context], \
            train_y, batch_size=8192, shuffle=True)
    print('Data generator initialized successfully.')

    ######################################################
    ## Model Structure
    ######################################################

    ## define the model
    def FunctionalDense(n, x, batchnorm=False, act='relu', lw1=0.0, dropout=None, name=''):
        if lw1 == 0.0:
            x = Dense(n, name=name+'_dense')(x)
        else:
            x = Dense(n, kernel_regularizer=l1(lw1), name=name+'_dense')(x)
        
        if batchnorm:
            x = BatchNormalization(name=name+'_batchnorm')(x)
            
        if act in {'relu', 'tanh', 'sigmoid'}:
            x = Activation(act, name=name+'_activation')(x)
        elif act =='prelu':
            x = PReLU(name=name+'_activation')(x)
        elif act == 'leakyrelu':
            x = LeakyReLU(name=name+'_activation')(x)
        elif act == 'elu':
            x = ELU(name=name+'_activation')(x)
        
        if dropout is not None and dropout > 0:
            x = Dropout(dropout, name=name+'_dropout')(x)
            
        return x

    def get_model(K, K0, lw=1e-4, lw1=1e-4, lr=1e-3, act='relu', batchnorm=False):
        embedding_inputs = []
        embedding_outputs = []
        for i in range(len(embedding_features) - 3):
            val_bound = 0.0 if i == 0 else 0.005
            tmp_input = Input(shape=(1,), dtype='int32', name=embedding_features[i]+'_input')
            tmp_embeddings = Embedding(int(train_embeddings[i].max()+1),
                    K if i == 0 else K0,
                    embeddings_initializer=RandomUniform(minval=-val_bound, maxval=val_bound),
                    embeddings_regularizer=l2(lw),
                    trainable=True,
                    name=embedding_features[i]+'_embeddings')(tmp_input)
            tmp_embeddings = Flatten(name=embedding_features[i]+'_flatten')(tmp_embeddings)
            
            embedding_inputs.append(tmp_input)
            embedding_outputs.append(tmp_embeddings)

        song_id_input = Input(shape=(1,), dtype='int32', name='song_id_input')
        before_song_id_input = Input(shape=(1,), dtype='int32', name='before_song_id_input')
        after_song_id_input = Input(shape=(1,), dtype='int32', name='after_song_id_input')

        embedding_inputs += [song_id_input, before_song_id_input, after_song_id_input]

        genre_inputs = []
        genre_outputs = []
        genre_embeddings = Embedding(int(np.max(train_genre)+1),
                K0,
                embeddings_initializer=RandomUniform(minval=-0.05, maxval=0.05),
                embeddings_regularizer=l2(lw),
                trainable=True,
                name='genre_embeddings')
        for i in range(len(genre_features)):
            tmp_input = Input(shape=(1,), dtype='int32', name=genre_features[i]+'_input')
            tmp_embeddings = genre_embeddings(tmp_input)
            tmp_embeddings = Flatten(name=genre_features[i]+'_flatten')(tmp_embeddings)
            
            genre_inputs.append(tmp_input)
            genre_outputs.append(tmp_embeddings)

        usr_input = Embedding(usr_feat.shape[0],
                usr_feat.shape[1],
                weights=[usr_feat],
                trainable=False,
                name='usr_feat')(embedding_inputs[0])
        usr_input = Flatten(name='usr_feat_flatten')(usr_input)
        
        song_input = Embedding(song_feat.shape[0],
                song_feat.shape[1],
                weights=[song_feat],
                trainable=False,
                name='song_feat')(song_id_input)
        song_input = Flatten(name='song_feat_flatten')(song_input)
        
        usr_component_input = Embedding(usr_component.shape[0],
                usr_component.shape[1],
                weights=[usr_component],
                trainable=False,
                name='usr_component')(embedding_inputs[0])
        usr_component_input = Flatten(name='usr_component_flatten')(usr_component_input)
        
        song_component_embeddings = Embedding(song_component.shape[0],
                song_component.shape[1],
                weights=[song_component],
                trainable=False,
                name='song_component')
        song_component_input = song_component_embeddings(song_id_input)
        song_component_input = Flatten(name='song_component_flatten')(song_component_input)
        before_song_component_input = song_component_embeddings(before_song_id_input)
        before_song_component_input = Flatten(name='before_song_component_flatten')(before_song_component_input)
        after_song_component_input = song_component_embeddings(after_song_id_input)
        after_song_component_input = Flatten(name='after_song_component_flatten')(after_song_component_input)
        
        usr_artist_component_input = Embedding(usr_artist_component.shape[0],
                usr_artist_component.shape[1],
                weights=[usr_artist_component],
                trainable=False,
                name='usr_artist_component')(embedding_inputs[0])
        usr_artist_component_input = Flatten(name='usr_artist_component_flatten')(usr_artist_component_input)
        
        song_artist_component_embeddings = Embedding(song_artist_component.shape[0],
                song_artist_component.shape[1],
                weights=[song_artist_component],
                trainable=False,
                name='song_artist_component')
        song_artist_component_input = song_artist_component_embeddings(song_id_input)
        song_artist_component_input = Flatten(name='song_artist_component_flatten')(song_artist_component_input)
        before_song_artist_component_input = song_artist_component_embeddings(before_song_id_input)
        before_song_artist_component_input = Flatten(name='before_song_artist_component_flatten')(before_song_artist_component_input)
        after_song_artist_component_input = song_artist_component_embeddings(after_song_id_input)
        after_song_artist_component_input = Flatten(name='after_song_artist_component_flatten')(after_song_artist_component_input)
        
        context_input = Input(shape=(len(context_features),), name='context_feat')
        
        # basic profiles
        usr_profile = concatenate(embedding_outputs[1:4]+[usr_input, \
                usr_component_input, usr_artist_component_input], name='usr_profile')
        song_profile = concatenate(embedding_outputs[4:7]+genre_outputs+[song_input, \
                song_component_input, song_artist_component_input], name='song_profile')
        before_song_profile = concatenate([before_song_component_input, before_song_artist_component_input], name='before_song_profile')
        after_song_profile = concatenate([after_song_component_input, after_song_artist_component_input], name='after_song_profile')
        
        # interaction
        usr_song = dot([embedding_outputs[0], embedding_outputs[4]], axes=1, normalize=False, name='usr_song')
        usr_before_song = dot([embedding_outputs[0], embedding_outputs[5]], axes=1, normalize=False, name='usr_before_song')
        usr_after_song = dot([embedding_outputs[0], embedding_outputs[6]], axes=1, normalize=False, name='usr_after_song')
        
        joint_embeddings = concatenate([usr_profile, song_profile, before_song_profile, after_song_profile, \
                usr_song, usr_before_song, usr_after_song, context_input], name='joint_embeddings')
        
        # top model
        preds0 = FunctionalDense(K*2, joint_embeddings, batchnorm=batchnorm, act=act, name='preds_0')
        preds1 = FunctionalDense(K*2, concatenate([joint_embeddings, preds0]), batchnorm=batchnorm, act=act, name='preds_1')
        preds2 = FunctionalDense(K*2, concatenate([joint_embeddings, preds0, preds1]), batchnorm=batchnorm, act=act, name='preds_2')
        
        preds = concatenate([joint_embeddings, preds0, preds1, preds2], name='prediction_aggr')
        preds = Dropout(0.5, name='prediction_dropout')(preds)
        preds = Dense(1, activation='sigmoid', name='prediction')(preds)
            
        model = Model(inputs=embedding_inputs+genre_inputs+[context_input], outputs=preds)
        opt = RMSprop(learning_rate=lr)

        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

        return model

    ######################################################
    ## Model Training
    ######################################################

    ## train the model
    para = pd.read_csv('./nn_record.csv').sort_values(by='val_auc', ascending=False)
    training_start_time = time.time()

    K = para['K'].values[config_index]
    K0 = para['K0'].values[config_index]
    lw = para['lw'].values[config_index]
    lw1 = para['lw1'].values[config_index]
    lr = para['lr'].values[config_index]
    lr_decay = para['lr_decay'].values[config_index]
    activation = para['activation'].values[config_index]
    batchnorm = para['batchnorm'].values[config_index]
    bst_epoch = para['bst_epoch'].values[config_index]
    train_loss0 = para['trn_loss'].values[config_index]

    while(True):
        print('K: %d, K0: %d, lw: %e, lw1: %e, lr: %e, lr_decay: %f, act: %s, batchnorm: %s'%(K, K0, lw, \
                lw1, lr, lr_decay, activation, batchnorm))
        
        model = get_model(K, K0, lw, lw1, lr, activation, batchnorm)

        lr_reducer = LearningRateScheduler(lambda x: lr*(lr_decay**x))
        
        # In TEST_MODE, use only 2 epochs for quick testing
        num_epochs = 2 if test_mode else bst_epoch
        if test_mode:
            print(f"TEST_MODE: Using {num_epochs} epochs instead of {bst_epoch}")
        
        hist = model.fit(train_flow, 
                 steps_per_epoch=train_flow.__len__(), 
                 epochs=num_epochs, 
                 callbacks=[lr_reducer])
        
        train_loss = hist.history['loss'][-1]
        
        os.makedirs('./temp_nn/models', exist_ok=True)
        model_path = './temp_nn/models/nn_model_seed%d_config%d.weights.h5' % (random_seed, config_index)
        model.save_weights(model_path)
        print(f'Model weights saved to: {model_path}')
        
        # Skip the loss check in test mode
        if test_mode:
            break
        
        if(train_loss < train_loss0 * 1.1):
            break
    
    training_end_time = time.time()
    training_runtime = training_end_time - training_start_time
    print(f'\nTraining runtime: {training_runtime:.2f} seconds ({training_runtime/60:.2f} minutes)')
    
    # Calculate validation metrics
    test_flow = dataGenerator.flow(test_embeddings + test_genre, [test_context], \
            batch_size=16384, shuffle=False)
    test_pred = model.predict(test_flow, 
                               steps=test_flow.__len__())
    
    # Calculate AUC on validation set
    val_auc_calculated = roc_auc_score(test_y.values, test_pred.ravel())
    val_loss = hist.history['loss'][-1]
    print('Model training done. Test set AUC (calculated): %.5f'%val_auc_calculated)
    
    # Create results dataframe with predictions and ground truth
    os.makedirs('./temp_nn', exist_ok=True)
    test_sub = pd.DataFrame({
        'id': test_id, 
        'prediction': test_pred.ravel(),
        'ground_truth_target': test_y.values
    })
    
    if test_mode:
        output_filename = './temp_nn/nn_%.5f_%.5f_seed%d_test.csv'%(val_auc_calculated, train_loss, random_seed)
    else:
        output_filename = './temp_nn/nn_%.5f_%.5f_seed%d.csv'%(val_auc_calculated, train_loss, random_seed)
    test_sub.to_csv(output_filename, index=False)
    
    print(f'Predictions saved to: {output_filename}')
    
    return {
        'seed': random_seed,
        'predictions': test_pred.ravel(),
        'ground_truth': test_y.values,
        'auc': val_auc_calculated,
        'loss': val_loss,
        'filename': output_filename
    }


######################################################
## Run Multiple Seeds
######################################################

if __name__ == '__main__':
    # Define seeds to use
    seeds = [13, 42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055]
    
    folder = 'training'
    
    if TEST_MODE:
        print("\n" + "="*60)
        print("TEST_MODE ENABLED - Using reduced data and epochs for quick testing")
        print("="*60 + "\n")
        # Use fewer seeds in test mode
        seeds = seeds[:2]  # Only use first 2 seeds for quick testing
    
    print(f'\n{"="*60}')
    print(f'Running training with {len(seeds)} different random seeds')
    print(f'{"="*60}\n')
    
    results = []
    
    for seed in seeds:
        try:
            result = train_nn_model(seed, folder, config_index=0, test_mode=TEST_MODE)
            results.append(result)
        except Exception as e:
            print(f'Error with seed {seed}: {str(e)}')
            import traceback
            traceback.print_exc()
            continue
    
    # Summary statistics
    print(f'\n{"="*60}')
    print('SUMMARY OF ALL RUNS')
    print(f'{"="*60}\n')
    
    if results:
        aucs = [r['auc'] for r in results]
        losses = [r['loss'] for r in results]
        
        summary_df = pd.DataFrame({
            'seed': [r['seed'] for r in results],
            'auc': aucs,
            'loss': losses,
            'filename': [r['filename'] for r in results]
        })
        
        print(summary_df.to_string(index=False))
        print(f'\nAUC Statistics:')
        print(f'  Mean: {np.mean(aucs):.5f}')
        print(f'  Std:  {np.std(aucs):.5f}')
        print(f'\nLoss Statistics:')
        print(f'  Mean: {np.mean(losses):.5f}')
        print(f'  Std:  {np.std(losses):.5f}')
        
        # Save summary
        summary_df.to_csv('./temp_nn/multi_seed_summary.csv', index=False)
        print(f'\nSummary saved to: ./temp_nn/multi_seed_summary.csv')
    else:
        print('No successful runs!')
