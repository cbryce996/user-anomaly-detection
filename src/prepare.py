import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

def read_csv_file(file_path):
    data = pd.read_csv(file_path)
    return data

def scale_and_normalize_data(data):
    scaler = StandardScaler()
    normalizer = Normalizer()
    data_s = scaler.fit_transform(data)
    data_n = normalizer.fit_transform(data_s)
    data_frame = pd.DataFrame(data_s, columns=scaler.get_feature_names_out(data.columns))
    return data_frame

def prepare_data(file_path):
    data = read_csv_file(file_path)
    prepared_data = scale_and_normalize_data(data)
    print (prepared_data)
    return prepared_data