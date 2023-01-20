import pandas as pd
from model import train_model,predict



if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    target_label = '_5s_side'
    train_model(data, target_label)
    pred, true_val = predict(data, target_label)
