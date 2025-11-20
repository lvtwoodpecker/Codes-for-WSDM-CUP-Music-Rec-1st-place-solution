import gc
import datetime
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Set TEST_MODE = True for quick testing (reduced data)
TEST_MODE = True  # Change to True for quick testing

RANDOM_SEED = 13
folder = 'training'

if TEST_MODE:
    print("\n" + "="*60)
    print("TEST_MODE ENABLED - Using reduced data for quick testing")
    print("="*60 + "\n")

#####################################################
## Data Loading
#####################################################

param_start_time = time.time()

print(f'Splitting data with random seed: {RANDOM_SEED}')
## load data
if folder == 'training':
    if TEST_MODE:
        print("TEST_MODE: Loading only 10,000 rows for quick testing")
        train_full = pd.read_csv('./input/%s/train_part.csv'%folder, nrows=10000)
        train_add_full = pd.read_csv('./input/%s/train_part_add.csv'%folder, nrows=10000)
    else:
        train_full = pd.read_csv('./input/%s/train_part.csv'%folder)
        train_add_full = pd.read_csv('./input/%s/train_part_add.csv'%folder)
elif folder == 'validation':
    if TEST_MODE:
        print("TEST_MODE: Loading only 10,000 rows for quick testing")
        train_full = pd.read_csv('./input/%s/train.csv'%folder, nrows=10000)
        train_add_full = pd.read_csv('./input/%s/train_add.csv'%folder, nrows=10000)
    else:
        train_full = pd.read_csv('./input/%s/train.csv'%folder)
        train_add_full = pd.read_csv('./input/%s/train_add.csv'%folder)

# Extract target before splitting
train_y_full = train_full['target'].copy()

# Split data 80-20
train_indices, test_indices, train_y, test_y = train_test_split(
    train_full.index, train_y_full, 
    test_size=0.2, 
    random_state=RANDOM_SEED,
    stratify=train_y_full  # maintain class distribution
)

# split train and train_add using the same indices
train = train_full.iloc[train_indices].reset_index(drop=True)
test = train_full.iloc[test_indices].reset_index(drop=True)
train_add = train_add_full.iloc[train_indices].reset_index(drop=True)
test_add = train_add_full.iloc[test_indices].reset_index(drop=True)

# reset indices for targets
train_y = train_y.reset_index(drop=True)
test_y = test_y.reset_index(drop=True)

# drop target from both train and test
train.drop(['target'], inplace=True, axis=1)
test.drop(['target'], inplace=True, axis=1)

print(f'Training set size: {len(train):,}')
print(f'Validation set size: {len(test):,}')

# create test_id for saving (using index)
test_id = test.index.values

train_add['source'] = train_add['source'].astype('category')
test_add['source'] = test_add['source'].astype('category')

cols = ['msno_artist_name_prob', 'msno_first_genre_id_prob', 'msno_xxx_prob', \
        'msno_language_prob', 'msno_yy_prob', 'source', 'msno_source_prob', \
        'song_source_system_tab_prob', 'song_source_screen_name_prob', \
        'song_source_type_prob']
for col in cols:
    train[col] = train_add[col].values
    test[col] = test_add[col].values

## merge data
member = pd.read_csv('./input/%s/members_gbdt.csv'%folder)

train = train.merge(member, on='msno', how='left')
test = test.merge(member, on='msno', how='left')

del member
gc.collect()

member_add = pd.read_csv('./input/%s/members_add.csv'%folder)

cols = ['msno', 'msno_song_length_mean', 'artist_msno_cnt']
train = train.merge(member_add[cols], on='msno', how='left')
test = test.merge(member_add[cols], on='msno', how='left')

del member_add
gc.collect()

song = pd.read_csv('./input/%s/songs_gbdt.csv'%folder)

train = train.merge(song, on='song_id', how='left')
test = test.merge(song, on='song_id', how='left')

cols = song.columns

song.columns = ['before_'+i for i in cols]
train = train.merge(song, on='before_song_id', how='left')
test = test.merge(song, on='before_song_id', how='left')

song.columns = ['after_'+i for i in cols]
train = train.merge(song, on='after_song_id', how='left')
test = test.merge(song, on='after_song_id', how='left')

del song
gc.collect()

print('Member/Song data loaded.')

#####################################################
## Additional Features
#####################################################

## contextual features
train['before_type_same'] = (train['before_source_type'] == train['source_type']) * 1.0
test['before_type_same'] = (test['before_source_type'] == test['source_type']) * 1.0

train['after_type_same'] = (train['after_source_type'] == train['source_type']) * 1.0
test['after_type_same'] = (test['after_source_type'] == test['source_type']) * 1.0

train['before_artist_same'] = (train['before_artist_name'] == train['artist_name']) * 1.0
test['before_artist_same'] = (test['before_artist_name'] == test['artist_name']) * 1.0

train['after_artist_same'] = (train['after_artist_name'] == train['artist_name']) * 1.0
test['after_artist_same'] = (test['after_artist_name'] == test['artist_name']) * 1.0
'''
train['timestamp_mean_diff'] = train['timestamp'] - train['msno_timestamp_mean']
test['timestamp_mean_diff'] = test['timestamp'] - test['msno_timestamp_mean']

train['timestamp_mean_diff_rate'] = train['timestamp_mean_diff'] / train['msno_timestamp_std']
test['timestamp_mean_diff_rate'] = test['timestamp_mean_diff'] / test['msno_timestamp_std']
'''
train['time_spent'] = train['timestamp'] - train['registration_init_time']
test['time_spent'] = test['timestamp'] - test['registration_init_time']

train['time_left'] = train['expiration_date'] - train['timestamp']
test['time_left'] = test['expiration_date'] - test['timestamp']
'''
train['msno_till_now_cnt_rate'] = train['msno_till_now_cnt'] - train['msno_rec_cnt']
test['msno_till_now_cnt_rate'] = test['msno_till_now_cnt'] - test['msno_rec_cnt']

train['msno_left_cnt'] = np.log1p(np.exp(train['msno_rec_cnt']) - \
        np.exp(train['msno_till_now_cnt']))
test['msno_left_cnt'] = np.log1p(np.exp(test['msno_rec_cnt']) - \
        np.exp(test['msno_till_now_cnt']))
'''
## user-side features
train['duration'] = train['expiration_date'] - train['registration_init_time']
test['duration'] = test['expiration_date'] - test['registration_init_time']

train['msno_upper_time'] = train['msno_timestamp_mean'] + train['msno_timestamp_std']
test['msno_upper_time'] = test['msno_timestamp_mean'] + test['msno_timestamp_std']

train['msno_lower_time'] = train['msno_timestamp_mean'] - train['msno_timestamp_std']
test['msno_lower_time'] = test['msno_timestamp_mean'] - test['msno_timestamp_std']

## song-side features
train['song_upper_time'] = train['song_timestamp_mean'] + train['song_timestamp_std']
test['song_upper_time'] = test['song_timestamp_mean'] + test['song_timestamp_std']

train['song_lower_time'] = train['song_timestamp_mean'] - train['song_timestamp_std']
test['song_lower_time'] = test['song_timestamp_mean'] - test['song_timestamp_std']

#####################################################
## Feature Processing
#####################################################

## set features to category
embedding_features = ['msno', 'city', 'gender', 'registered_via', \
        'song_id', 'artist_name', 'composer', 'lyricist', 'language', \
        'first_genre_id', 'second_genre_id', 'third_genre_id', 'cc', 'xxx', \
        'isrc_missing', 'source_system_tab', 'source_screen_name', 'source_type']
song_id_feat = ['artist_name', 'composer', 'lyricist', 'language', \
        'first_genre_id', 'second_genre_id', 'third_genre_id', 'cc', 'xxx', \
        'isrc_missing']
embedding_features += ['before_'+i for i in song_id_feat]
embedding_features += ['after_'+i for i in song_id_feat]
embedding_features += ['before_song_id', 'after_song_id', 'before_source_type', \
        'after_source_type', 'before_type_same', 'after_type_same', \
        'before_artist_same', 'after_artist_same']

for feat in embedding_features:
    train[feat] = train[feat].astype('category')
    test[feat] = test[feat].astype('category')

## feature selection
feat_importance = pd.read_csv('./lgb_feature_importance.csv')
feature_name = feat_importance['name'].values
feature_importance = feat_importance['importance'].values

drop_col = feature_name[feature_importance<85]
def transfer(x):
    if x == 'msno_source_screen_name_15':
        return 'msno_source_screen_name_17'
    elif  x == 'msno_source_screen_name_16':
        return 'msno_source_screen_name_18'
    elif  x == 'msno_source_screen_name_17':
        return 'msno_source_screen_name_19'
    elif  x == 'msno_source_screen_name_18':
        return 'msno_source_screen_name_20'
    elif  x == 'msno_source_screen_name_19':
        return 'msno_source_screen_name_21'
    elif  x == 'msno_source_screen_name_20':
        return 'msno_source_screen_name_22'
    else:
        return x
drop_col = [transfer(i) for i in drop_col]

# Only drop columns that actually exist in the dataframes
existing_drop_col = [col for col in drop_col if col in train.columns]
train.drop(existing_drop_col, axis=1, inplace=True)
test.drop(existing_drop_col, axis=1, inplace=True)

## print data information
print('Data preparation done.')
print('Training data shape:')
print(train.shape)
print('Testing data shape:')
print(test.shape)
print('Features involved:')
print(train.columns)
param_end_time = time.time()
param_runtime = param_end_time - param_start_time
print(f'\nData preparation runtime: {param_runtime:.2f} seconds ({param_runtime/60:.2f} minutes)')

#####################################################
## Model Training
#####################################################

## model training
training_start_time = time.time()
train_data = lgb.Dataset(train, label=train_y, free_raw_data=True)

del train
gc.collect()

para = pd.read_csv('./lgb_record.csv').sort_values(by='val_auc', ascending=False)
for i in range(1):
    params = {
        'boosting_type': para['type'].values[i],
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'], 
        
        'learning_rate': para['lr'].values[i],
        
        'num_leaves': para['n_leaf'].values[i],
        'max_depth': para['n_depth'].values[i],
        'min_data_in_leaf': para['min_data'].values[i],
        
        'feature_fraction': para['feature_frac'].values[i],
        'bagging_fraction': para['bagging_frac'].values[i],
        'bagging_freq': para['bagging_freq'].values[i],
        
        'lambda_l1': para['l1'].values[i],
        'lambda_l2': para['l2'].values[i],
        'min_gain_to_split': para['min_gain'].values[i],
        'min_sum_hessian_in_leaf': para['hessian'].values[i],
        
        'num_threads': 16,
        'verbose': -1,
        'is_training_metric': 'True'
    }
    
    print('Hyper-parameters:')
    print(params)

    num_round = para['bst_rnd'].values[i]
    
    # In TEST_MODE, use fewer rounds for quick testing
    if TEST_MODE:
        num_round = min(10, num_round)  # Use max 10 rounds in test mode
        print(f'TEST_MODE: Using {num_round} rounds instead of {para["bst_rnd"].values[i]}')
    else:
        print('Round number: %d'%num_round)

    gbm = lgb.train(params, train_data, num_round, valid_sets=[train_data], callbacks=[lgb.log_evaluation(100)])

    training_end_time = time.time()
    training_runtime = training_end_time - training_start_time
    print(f'\n Training runtime: {training_runtime:.2f} seconds ({training_runtime/60:.2f} minutes)')

    feature_importance = pd.DataFrame({'name':gbm.feature_name(), 'importance':gbm.feature_importance()}).sort_values(by='importance', ascending=False)
    feature_importance.to_csv('./feat_importance_for_test_seed%d.csv'%(RANDOM_SEED), index=False)
       
    test_pred = gbm.predict(test)
    
    val_auc_calculated = roc_auc_score(test_y.values, test_pred)
    print('Model training done. Test set AUC (calculated): %.5f'%val_auc_calculated)
    
    # Create results dataframe with predictions and ground truth
    test_sub = pd.DataFrame({
        'id': test_id, 
        'prediction': test_pred,
        'ground_truth_target': test_y.values
    })
    
    if TEST_MODE:
        output_filename = './submission/lgb_%.5f_seed%d_test.csv'%(val_auc_calculated, RANDOM_SEED)
    else:
        output_filename = './submission/lgb_%.5f_seed%d.csv'%(val_auc_calculated, RANDOM_SEED)
    test_sub.to_csv(output_filename, index=False)
    
    print(f'Predictions saved to: {output_filename}')    
