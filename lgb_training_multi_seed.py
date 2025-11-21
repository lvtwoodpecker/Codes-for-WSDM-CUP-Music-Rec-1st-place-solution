import gc
import datetime
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import os

# Set TEST_MODE = True for quick testing (fewer rounds, smaller data)
TEST_MODE = False  # Change to True for quick testing

def train_lgb_model(random_seed, folder='training', test_mode=False):
    """
    Train LightGBM model with a specific random seed for train/val split.
    Returns predictions, ground truth, and metrics.
    """
    print(f'\n{"="*60}')
    print(f'Training with random seed: {random_seed}')
    print(f'{"="*60}\n')
    
    param_start_time = time.time()
    
    ## Define dtypes for memory optimization
    dtypes_train = {
        'target': 'int8',  # Binary target, int8 saves memory
        'msno': 'category', 
        'song_id': 'category',
        'before_song_id': 'category',
        'after_song_id': 'category'
    }
    
    ## load data
    if test_mode:
        print("TEST_MODE: Loading only 10,000 rows for quick testing")
        nrows = 10000
    else:
        nrows = None
    
    if folder == 'training':
        if nrows:
            train_full = pd.read_csv('./input/%s/train_part.csv'%folder, nrows=nrows, dtype=dtypes_train)
            train_add_full = pd.read_csv('./input/%s/train_part_add.csv'%folder, nrows=nrows)
        else:
            train_full = pd.read_csv('./input/%s/train_part.csv'%folder, dtype=dtypes_train)
            train_add_full = pd.read_csv('./input/%s/train_part_add.csv'%folder)
    elif folder == 'validation':
        if nrows:
            train_full = pd.read_csv('./input/%s/train.csv'%folder, nrows=nrows, dtype=dtypes_train)
            train_add_full = pd.read_csv('./input/%s/train_add.csv'%folder, nrows=nrows)
        else:
            train_full = pd.read_csv('./input/%s/train.csv'%folder, dtype=dtypes_train)
            train_add_full = pd.read_csv('./input/%s/train_add.csv'%folder)

    # Extract target before splitting
    train_y_full = train_full['target'].copy()
    
    # Convert msno to numeric type for merging (category/object causes merge issues)
    # We read as category for memory, but convert to int64 for merging compatibility
    if train_full['msno'].dtype.name in ['category', 'object']:
        train_full['msno'] = pd.to_numeric(train_full['msno'], errors='coerce').astype('int64')
    elif train_full['msno'].dtype.name != 'int64':
        train_full['msno'] = train_full['msno'].astype('int64')

    # Split data 80-20
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

    # reset indices for targets
    train_y = train_y.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)

    # drop target from both train and test
    train.drop(['target'], inplace=True, axis=1)
    test.drop(['target'], inplace=True, axis=1)

    print(f'Training set size: {len(train):,}')
    print(f'Validation set size: {len(test):,}')

    # Save original indices BEFORE reset_index for mapping back to train_part.csv
    # This allows us to map predictions back to original data rows
    original_test_indices = test_indices.copy()
    
    # create test_id for saving (using sequential index for now, but we'll save original too)
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

    train['time_spent'] = train['timestamp'] - train['registration_init_time']
    test['time_spent'] = test['timestamp'] - test['registration_init_time']

    train['time_left'] = train['expiration_date'] - train['timestamp']
    test['time_left'] = test['expiration_date'] - test['timestamp']

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
    print('Training data shape:', train.shape)
    print('Testing data shape:', test.shape)

    feat_cnt = train.shape[1]
    
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
    
    params = {
        'boosting_type': para['type'].values[0],
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'], 
        
        'learning_rate': para['lr'].values[0],
        
        'num_leaves': para['n_leaf'].values[0],
        'max_depth': para['n_depth'].values[0],
        'min_data_in_leaf': para['min_data'].values[0],
        
        'feature_fraction': para['feature_frac'].values[0],
        'bagging_fraction': para['bagging_frac'].values[0],
        'bagging_freq': para['bagging_freq'].values[0],
        
        'lambda_l1': para['l1'].values[0],
        'lambda_l2': para['l2'].values[0],
        'min_gain_to_split': para['min_gain'].values[0],
        'min_sum_hessian_in_leaf': para['hessian'].values[0],
        
        'num_threads': 16,
        'verbose': -1,
        'is_training_metric': 'True'
    }
    
    print('Hyper-parameters loaded from best model.')

    num_round = para['bst_rnd'].values[0]
    # In TEST_MODE, use only 10 rounds for quick testing
    if test_mode:
        num_round = min(10, num_round)
        print(f'TEST_MODE: Using {num_round} rounds instead of {para["bst_rnd"].values[0]}')
    else:
        print(f'Round number: {num_round}')

    gbm = lgb.train(params, train_data, num_round, valid_sets=[train_data], callbacks=[lgb.log_evaluation(100)])

    training_end_time = time.time()
    training_runtime = training_end_time - training_start_time
    print(f'\nTraining runtime: {training_runtime:.2f} seconds ({training_runtime/60:.2f} minutes)')

    # Make predictions on validation set
    test_pred = gbm.predict(test)
    
    # Calculate metrics
    val_auc = roc_auc_score(test_y.values, test_pred)
    val_loss = log_loss(test_y.values, test_pred)
    
    print(f'Test set AUC (calculated): {val_auc:.5f}')
    
    # Create results dataframe with predictions and ground truth
    # Include original_index to map back to train_part.csv, and song_id/msno for artist mapping
    # Note: song_id and msno are category type, convert back to int64 for saving
    test_sub = pd.DataFrame({
        'id': test_id,  # Sequential index in validation set (0, 1, 2, ...)
        'original_index': original_test_indices,  # original row index from train_part.csv
        'song_id': test['song_id'].astype('int64').values,
        'msno': test['msno'].astype('int64').values,  # Member ID
        'prediction': test_pred,
        'ground_truth_target': test_y.values
    })
    
    # Save to CSV
    os.makedirs('./temp_lgb', exist_ok=True)
    if test_mode:
        output_filename = './temp_lgb/lgb_%.5f_seed%d_test.csv'%(val_auc, random_seed)
    else:
        output_filename = './temp_lgb/lgb_%.5f_seed%d.csv'%(val_auc, random_seed)
    test_sub.to_csv(output_filename, index=False)
    
    print(f'Predictions saved to: {output_filename}')
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'name': gbm.feature_name(), 
        'importance': gbm.feature_importance()
    }).sort_values(by='importance', ascending=False)
    feature_importance.to_csv('./feat_importance_for_test_seed%d.csv'%(random_seed), index=False)
    
    # Save summary file for easy recovery
    seed_summary_filename = './temp_lgb/seed%d_summary.csv' % random_seed
    seed_summary = pd.DataFrame({
        'seed': [random_seed],
        'auc': [val_auc],
        'loss': [val_loss],
        'filename': [output_filename]
    })
    seed_summary.to_csv(seed_summary_filename, index=False)
    
    return {
        'seed': random_seed,
        'predictions': test_pred,
        'ground_truth': test_y.values,
        'auc': val_auc,
        'loss': val_loss,
        'filename': output_filename
    }


#####################################################
## Run Multiple Seeds
#####################################################

if __name__ == '__main__':
    # Define seeds to use
    seeds = [13, 25, 37, 49, 61, 73, 85, 97, 109, 121]
    
    folder = 'training'
    
    if TEST_MODE:
        print("\n" + "="*60)
        print("TEST_MODE ENABLED - Using reduced data and rounds for quick testing")
        print("="*60 + "\n")
        # Use fewer seeds in test mode
        seeds = seeds[:2]  # Only use first 2 seeds for quick testing
    
    print(f'\n{"="*60}')
    print(f'Running training with {len(seeds)} different random seeds')
    print(f'{"="*60}\n')
    
    results = []
    os.makedirs('./temp_lgb', exist_ok=True)
    
    for seed in seeds:
        # Check if this seed has already been completed by looking for summary file
        seed_summary_file = './temp_lgb/seed%d_summary.csv' % seed
        
        if os.path.exists(seed_summary_file):
            print(f'\nSeed {seed} already completed - loading from summary file\n')
            try:
                # Load existing summary
                seed_summary = pd.read_csv(seed_summary_file)
                existing_filename = seed_summary['filename'].values[0]
                
                # Verify the actual output file exists
                if os.path.exists(existing_filename):
                    # Load the predictions file to reconstruct results
                    existing_results = pd.read_csv(existing_filename)
                    
                    result = {
                        'seed': seed,
                        'predictions': existing_results['prediction'].values,
                        'ground_truth': existing_results['ground_truth_target'].values,
                        'auc': seed_summary['auc'].values[0],
                        'loss': seed_summary['loss'].values[0],
                        'filename': existing_filename
                    }
                    results.append(result)
                    continue
                else:
                    print(f'Warning: Summary file exists but output file {existing_filename} not found. Re-running seed {seed}.')
            except Exception as e:
                print(f'Warning: Could not load existing results for seed {seed}: {e}')
                print(f'Re-running seed {seed}...')
        
        # Run training for this seed
        try:
            result = train_lgb_model(seed, folder, test_mode=TEST_MODE)
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
        summary_df.to_csv('./temp_lgb/multi_seed_summary.csv', index=False)
        print(f'\nSummary saved to: ./temp_lgb/multi_seed_summary.csv')
    else:
        print('No successful runs!')

