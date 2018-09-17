# Kaggle task: https://www.kaggle.com/c/demand-forecasting-kernels-only/data
import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.utils import Sequence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.externals import joblib
from shutil import copyfile
import os
import sys
from pathlib import Path
import logging
import boto
import boto.s3.connection
from boto.s3.key import Key
import argparse
import datetime as dt

# AWS params to save results
aws_access_key = ''
aws_secret_key = ''
aws_bucket_name = ''
aws_experiment_dir = 'ml_results/lstm'
aws_creds_file_name = 'aws_credentials.txt'

# Path properties
train_path = '../data/train.csv'
train_strat_path = '../data/train_strat.csv'
val_strat_path = '../data/val_strat.csv'
save_path = 'save/'
output_log_file_name = 'output.log'
best_model_file_name = 'model.best.hdf5'
final_model_file_name = 'model.hdf5'
python_file_dir = os.path.dirname(os.path.abspath(__file__))

# Model properties
experiment_description = 'Description: Basic LSTM model. Predict one day at a time.'
batch_size = 1
epochs = 25
steps_per_epoch = 1000
steps_per_epoch_val = 200
input_length = 50
days_to_predict = 90

# Calculated fields
encoders = {}
scalers = {}
features = pd.DataFrame()


# Load data
def read_train_data():
    train = pd.read_csv(train_path, nrows=2500)
    return train

def split_to_val_test(data):
    train, val = train_test_split(data, shuffle=False, train_size=0.8, test_size=0.2)

    return train, val

def split_to_train_val_statified(data):
    train_file = Path(train_strat_path)
    val_file = Path(val_strat_path)
    if train_file.is_file() and val_file.is_file():
        train_data = pd.read_csv(train_strat_path)
        val_data = pd.read_csv(val_strat_path)
        return train_data, val_data

    print("Starting to stratified split data to train and valiation.")
    groups = data.groupby(['store', 'item'])
    train_size = 0.8

    train_data = pd.DataFrame(columns=data.columns.values)
    val_data = pd.DataFrame(columns=data.columns.values)
    for key, group in groups:
        group_size = len(group)
        train_length = int(group_size * train_size)

        group_train = group[0:train_length]
        train_data = train_data.append(group_train, ignore_index=True)

        group_test = group[train_length:]
        val_data = val_data.append(group_test, ignore_index=True)

    train_data.to_csv(train_strat_path, index=False)
    val_data.to_csv(val_strat_path, index=False)

    return train_data, val_data


# Analyze
def print_date_ranges(train, val):
    min_train = train['date'].min()
    max_train = train['date'].max()
    min_val = val['date'].min()
    max_val = val['date'].max()
    print('Train range {0:%Y %m %d} - {1:%Y %m %d}' \
          .format(min_train, max_train))
    print('Validation range {0:%Y %m %d} - {1:%Y %m %d}' \
          .format(min_val, max_val))

def describe_data(train):
    print(train.head())
    print(train.columns)
    print(train.describe())
    print('Unique store ids {}'.format(train['store'].unique()))

    print('Groupped by storeid:')
    groups = train.groupby(by='store')
    for group in groups:
        print('Group {0} has {1} items'.format(group[0], group[1].size))

    min_train = train['date'].min()
    max_train = train['date'].max()
    print('Data range {0:%Y %m %d} - {1:%Y %m %d}' \
          .format(min_train, max_train))


# Prepare data
def create_features(data):
    data['date'] = data['date'].astype('datetime64')
    data['day'] = pd.Series(data['date']).dt.dayofyear
    data['year'] = pd.Series(data['date']).dt.year

def encode_data(train, val):
    one_hot_encoder_fit(train, val, 'store')
    one_hot_encoder_fit(train, val, 'item')

    scale_column(train, val, 'sales')
    scale_column(train, val, 'day')
    scale_column(train, val, 'year')

def one_hot_encoder_fit(train_data, val_data, column):
    # fit on train data only
    encoders[column] = OneHotEncoder(sparse=False, dtype=np.int64)
    encoders[column].fit(train[column].values.reshape(-1, 1))
    print('%s values: %s' % (column, encoders[column].active_features_))

def scale_column(train_data, val_data, column):
    scalers[column] = StandardScaler()
    scaler = scalers[column]

    values = train_data[column].values
    values = values.reshape((len(values), 1))
    scaler = scaler.fit(values)
    print('%s Mean: %f, StandardDeviation: %f' % (column, scaler.mean_, math.sqrt(scaler.var_)))

    train_data[column] = scaler.transform(values)

    val_values = val_data[column].values
    val_values = val_values.reshape((len(val_values), 1))
    val[column] = scaler.transform(val_values)

def count_features():
    global features

    features = pd.DataFrame(columns=['name', 'length', 'index_start', 'index_end'])
    register_feature('store', len(encoders['store'].active_features_))
    register_feature('sales', 1)
    register_feature('day', 1)
    register_feature('year', 1)
    register_feature('item', len(encoders['item'].active_features_))
    features = features.set_index('name')

    print('Feature names registered: %s' % features.index.values)
    print('Feature array length: %s' % features['index_end'].max())

def register_feature(name, length):
    global features

    index_start = features['index_end'].max() if len(features) > 0 else 0

    features = features.append({
        'name': name,
        'length': length,
        'index_start': index_start,
        'index_end': index_start + length
    }, ignore_index=True)

class SalesSequence(Sequence):
    def __init__(self, data, batch_size, input_len, output_len):
        self.batch_size = batch_size
        self.input_len = input_len
        self.output_len = output_len

        groups_grouping = data.groupby(['store', 'item'])
        self.groups = [v for k, v in groups_grouping]

        group_lengths = [int(np.ceil((len(x) - input_len - output_len) / self.batch_size)) for x in self.groups]
        self.group_ranges = np.cumsum(group_lengths)

    def __len__(self):
        range_end = self.group_ranges[-1]
        return range_end

    def __getitem__(self, idx):
        X = list()
        y = list()

        for b_i in range(self.batch_size):
            group_index = next(i for i,v in enumerate(self.group_ranges) if v > idx)
            group = self.groups[group_index]
            group_range_start = 0 if group_index == 0 else self.group_ranges[group_index - 1]

            input_start = (idx - group_range_start) * self.batch_size
            input_end = input_start + self.input_len
            b_X = group[input_start: input_end]
            b_X = prepare_input_features(b_X)
            X.append(b_X)

            output_start = input_end
            output_end = output_start + self.output_len
            b_y = group.iloc[output_start:output_end, 3].values.reshape(self.output_len, 1)
            y.append(b_y)

            idx += 1
            range_end = self.group_ranges[-1]
            if idx >= range_end:
                break

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        return X, y

def prepare_input_features(sequence):
    features_max_index = features['index_end'].max()
    store_start, store_end = features.at['store', 'index_start'], features.at['store', 'index_end']
    sales_start, sales_end = features.at['sales', 'index_start'], features.at['sales', 'index_end']
    day_start, day_end = features.at['day', 'index_start'], features.at['day', 'index_end']
    year_start, year_end = features.at['year', 'index_start'], features.at['year', 'index_end']
    item_start, item_end = features.at['item', 'index_start'], features.at['item', 'index_end']

    X = np.zeros((1, input_length, features_max_index), dtype=np.float64)
    X[0, :, store_start:store_end] = encoders['store'].transform(sequence['store'].values.reshape(-1, 1))
    X[0, :, sales_start:sales_end] = sequence['sales'].values.reshape(-1, 1)
    X[0, :, day_start:day_end] = sequence['day'].values.reshape(-1, 1)
    X[0, :, year_start:year_end] = sequence['year'].values.reshape(-1, 1)
    X[0, :, item_start:item_end] = encoders['item'].transform(sequence['item'].values.reshape(-1, 1))

    return X

def append_features_to_sequence(X, sales):
    features_max_index = features['index_end'].max()
    store_start, store_end, store_length = features.at['store', 'index_start'], features.at['store', 'index_end'], features.at['store', 'length']
    sales_start, sales_end = features.at['sales', 'index_start'], features.at['sales', 'index_end']
    day_start, day_end = features.at['day', 'index_start'], features.at['day', 'index_end']
    year_start, year_end = features.at['year', 'index_start'], features.at['year', 'index_end']
    item_start, item_end = features.at['item', 'index_start'], features.at['item', 'index_end']

    day_scaled = X[0, -1, day_start].reshape(1, 1)
    day = scalers['day'].inverse_transform(day_scaled)
    year_scaled = X[0, -1, year_start].reshape(1, 1)
    year = scalers['year'].inverse_transform(year_scaled)
    next_date = dt.date.fromordinal(dt.date(year, 1, 1).toordinal() + day - 1)
    next_date = next_date + dt.timedelta(days=1)
    next_doy = next_date.timetuple().tm_yday
    next_year = next_date.year

    new_row = np.zeros((1, 1, features_max_index), dtype=np.float64)
    new_row[0, 0, store_start:store_end] = X[0][0][store_start:store_end]
    new_row[0][0][sales_start] = sales
    new_row[0][0][day_start] = scalers['day'].transform(next_doy)
    new_row[0][0][year_start] = scalers['year'].transform(next_year)
    new_row[0, 0, item_start:item_end] = X[0][0][item_start:item_end]
    new_array = np.append(X, new_row, axis=1)
    return new_array


# Model
def define_model():
    features_count = features['index_end'].max()

    # define model
    model = Sequential()
    model.add(LSTM(units=1024, input_shape=(input_length, features_count), stateful=False))
    model.add(Dense(512))
    model.add(Dense(128))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def load_model_weights(model):
    #scaler = joblib.load(scaler_filename)

    filepath = save_path + final_model_file_name
    if os.path.isfile(filepath) == False:
        print('No model weights found in %s' % filepath)
        return

    model.load_weights(filepath)
    print('Loaded model weights from %s' % filepath)

def train_model(model, train, val):
    print("Starting model training")
    sequences = SalesSequence(train, batch_size, input_length, 1)
    val_sequences = SalesSequence(val, batch_size, input_length, 1)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0,
                                   verbose=0, mode='auto')

    filepath = save_path + best_model_file_name
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

    history = model.fit_generator(sequences, verbose=1, epochs=epochs
                      , steps_per_epoch=steps_per_epoch, validation_steps=steps_per_epoch_val
                      , validation_data=val_sequences, callbacks=[early_stopping, checkpoint])

    return history

def evaluate_model(model, val):
    print("Starting evaluation for %d days predictions" % days_to_predict)

    # prepare data
    val_sequences = SalesSequence(val, 1, input_length, days_to_predict)
    max_id = val_sequences.__len__()
    rand_ids = np.random.randint(0, max_id, steps_per_epoch_val)
    progress(0, len(rand_ids), 'Evaluation')

    # evaluate model
    yhat_results = list()
    y_results = list()

    for sample_index in range(len(rand_ids)):
        rand_id = rand_ids[sample_index]
        X, y = val_sequences.__getitem__(rand_id)
        yhat = predict_next_n_days(model, X)
        yhat_inversed = scalers['sales'].inverse_transform(yhat)
        yhat_inversed = yhat_inversed.reshape(yhat_inversed.shape[0])
        yhat_results.append(yhat_inversed)

        y_inversed = scalers['sales'].inverse_transform(y)
        y_inversed = y_inversed.reshape(y_inversed.shape[0])
        y_results.append(y_inversed)
        progress(sample_index + 1, len(rand_ids), 'Evaluation') #, prefix='Progress:', suffix='Complete', length=50)

    mse_score = mean_squared_error(y_results, yhat_results)
    print('Mean squared error: %f' % mse_score)
    return mse_score

def predict_next_n_days(model, X):
    yhat_results = list()
    input_sequence = X

    for _ in range(days_to_predict):
        yhat = model.predict(input_sequence)
        yhat_results.append(yhat[0])

        input_sequence = input_sequence[:,1:]
        input_sequence = append_features_to_sequence(input_sequence, yhat[0][0])

    return yhat_results

def sanity_check(model):
    print("Starting sanity check")

    val_sequences = SalesSequence(val, 1, input_length, days_to_predict)
    X, y = val_sequences.__getitem__(0)

    yhat = predict_next_n_days(model, X)
    yhat_inversed = scalers['sales'].inverse_transform(yhat)
    yhat_inversed = yhat_inversed.reshape(yhat_inversed.shape[0])

    y_inversed = scalers['sales'].inverse_transform(y)
    y_inversed = y_inversed.reshape(y_inversed.shape[0])

    sanity_check_results = pd.DataFrame(data=yhat_inversed, columns=['predicted'])
    sanity_check_results['actual'] = y_inversed
    return sanity_check_results


# Print iterations progress
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    print('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


# Save artifacts
class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """
   def __init__(self, std_output, logger, log_level=logging.INFO):
      self.std_output = std_output
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''

   def write(self, buf):
      self.std_output.write(buf)
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())

   def flush(self):
      self.std_output.flush()

def setup_logging():
    if os.path.isfile(output_log_file_name):
        os.remove(output_log_file_name)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        filename=output_log_file_name,
        filemode='w'
    )

    std_output = sys.stdout
    stdout_logger = logging.getLogger('STDOUT')
    sl = StreamToLogger(std_output, stdout_logger, logging.INFO)
    sys.stdout = sl

    std_err = sys.stderr
    stderr_logger = logging.getLogger('STDERR')
    sl = StreamToLogger(std_err, stderr_logger, logging.ERROR)
    sys.stderr = sl

def create_save_dir():
    real_save_dir = os.path.realpath(save_path)
    if not os.path.exists(real_save_dir):
        os.makedirs(real_save_dir)

def save_results(model, score, history, train_duration, train_len, val_len, sanity_check):
    # Model serialized
    model_path = save_path + final_model_file_name
    tf.keras.models.save_model(model,
        filepath=model_path, overwrite=True, include_optimizer=True)

    # Scalers
    for key in scalers:
        scaler_path = save_path + key + '_scaler.pkl'
        joblib.dump(scalers[key], scaler_path)

    # Encoders
    for key in encoders:
        encoder_path = save_path + key + '_ohe.pkl'
        joblib.dump(encoders[key], encoder_path)

    # Description
    description_path = save_path + 'description.txt'
    pd.set_option('display.precision', 2)
    with open(description_path, 'w') as f:
        f.write(experiment_description)
        f.write('Score on %d days prediction: %d\n' % (days_to_predict, score))
        f.write('\n')
        f.write('Epoch, Loss, Validation Loss:\n')
        for epoch in range(0, len(history.history['loss'])):
            f.write('Epoch %d - %0.4f %0.4f\n' % (epoch + 1, history.history['loss'][epoch], history.history['val_loss'][epoch]))
        f.write('\n')
        f.write('Train duration: {0}\n'.format(train_duration))
        f.write('Train rows number: {0}\n'.format(train_len))
        f.write('Validation rows number: {0}\n'.format(val_len))
        f.write('\n')
        f.write('Model summary:\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write('\n')
        f.write('Model parameters:\n')
        for k, v in history.params.items():
            f.write('%s=%s\n' % (k, v))
        f.write('Batch size=%s\n' % batch_size)
        f.write('Steps per training epoch=%s\n' % steps_per_epoch)
        f.write('Steps per validation=%s\n' % steps_per_epoch_val)
        f.write('Input length=%s\n' % input_length)
        f.write('Days to predict=%s\n' % days_to_predict)
        f.write('\n\n')
        f.write('Sanity check prediction:\n')
        f.write(sanity_check.to_string())

    # Python files
    py_files = [f for f in os.listdir(python_file_dir)
                if os.path.isfile(os.path.join(python_file_dir, f))
                and f.endswith(".py")]
    for f in py_files:
        file_src = os.path.join(python_file_dir, f)
        file_dest = save_path + f
        copyfile(file_src, file_dest)

    # Output logs
    log_src = os.path.join(python_file_dir, output_log_file_name)
    log_dest = save_path + output_log_file_name
    copyfile(log_src, log_dest)

def read_aws_creds():
    global aws_access_key
    global aws_secret_key
    global aws_bucket_name

    experiment_aws_cred_dir = os.path.join(python_file_dir, aws_creds_file_name)
    runner_aws_cred_dir = os.path.join(os.path.dirname(python_file_dir), aws_creds_file_name)
    cred_read = read_aws_creds_file(experiment_aws_cred_dir)
    if cred_read:
        return
    cred_read = read_aws_creds_file(runner_aws_cred_dir)
    if cred_read:
        return

    parser = argparse.ArgumentParser(description='Run machine learning experiment.')
    parser.add_argument('--aws_access', metavar='YOUR_AWS_ACCESS_KEY', type=str, required=True,
                        help='Your AWS access key to store results in S3')
    parser.add_argument('--aws_secret', metavar='YOUR_AWS_SECRET_KEY', type=str, required=True,
                        help='Your AWS secret key to store results in S3')
    parser.add_argument('--aws_bucket', metavar='YOUR_AWS_BUCKET_NAME', type=str, required=True,
                        help='Your AWS S3 bucket name to store results')

    args = parser.parse_args()
    aws_access_key = args.aws_access
    aws_secret_key = args.aws_secret
    aws_bucket_name = args.aws_bucket

def read_aws_creds_file(path):
    global aws_access_key
    global aws_secret_key
    global aws_bucket_name

    try:
        with open(path, 'r') as f:
            aws_access_key = f.readline().rstrip('\n')
            aws_secret_key = f.readline().rstrip('\n')
            aws_bucket_name = f.readline().rstrip('\n')
    except Exception as e:
        msg = 'Exception while reading aws cred file: {}'.format(sys.exc_info()[0])
        print(msg, file=sys.stderr)

    has_empty_creds = aws_access_key == '' \
                or aws_secret_key == '' \
                or aws_bucket_name == ''
    completed = not has_empty_creds
    return completed

def copy_results_to_s3():
    conn = boto.connect_s3(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    bucket = conn.get_bucket(aws_bucket_name)

    time_formatted = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    experiment_key_start = '{0}/{1}'.format(aws_experiment_dir, time_formatted)

    save_dir = os.path.join(python_file_dir, save_path)
    artifact_files = [f for f in os.listdir(save_dir)
                if os.path.isfile(os.path.join(save_dir, f))]
    for f in artifact_files:
        k = Key(bucket)
        k.key = '{}/{}'.format(experiment_key_start, f)
        file_src = os.path.join(save_dir, f)
        k.set_contents_from_filename(file_src)

    print("Uploaded results to S3 %s path" % aws_experiment_dir)


# Entry point
if __name__ == "__main__":
    setup_logging()
    read_aws_creds()
    create_save_dir()

    data = read_train_data()
    train, val = split_to_train_val_statified(data)
    create_features(train)
    create_features(val)
    print_date_ranges(train, val)

    train_start_time_utc = dt.datetime.utcnow()
    encode_data(train, val)
    count_features()
    model = define_model()
    load_model_weights(model)
    history = train_model(model, train, val)
    train_duration = dt.datetime.utcnow() - train_start_time_utc

    score = evaluate_model(model, val)
    sanity_check_results = sanity_check(model)
    save_results(model, score, history, train_duration, len(train), len(val), sanity_check_results)
    copy_results_to_s3()