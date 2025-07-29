import logging
import os
from datetime import datetime
from glob import glob

import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')

def load_latest_model():
    """Загружает последнюю сохраненную модель"""
    model_files = glob(f'{path}/data/models/cars_pipe_*.pkl')
    latest_model = max(model_files, key=os.path.getctime)
    
    with open(latest_model, 'rb') as f:
        model = dill.load(f)
    
    logging.info(f'Model loaded: {latest_model}')
    return model

def load_test_data():
    """Загружает все тестовые данные из папки data/test"""
    test_files = glob(f'{path}/data/test/*.json')
    dfs = []
    
    for file in test_files:
        try:
            df = pd.read_json(file, orient='index').T  
            dfs.append(df)
        except ValueError as e:
            df = pd.read_json(file)
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No test files found in data/test directory")
    
    return pd.concat(dfs, ignore_index=True)


def make_predictions(model, data):
    """Делает предсказания на данных"""
    return model.predict(data)


def save_predictions(predictions):
    """Сохраняет предсказания в файл"""
    pred_df = pd.DataFrame(predictions, columns=['predicted_price_category'])
    os.makedirs(f'{path}/data/predictions', exist_ok=True)
    
    pred_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    pred_df.to_csv(pred_filename, index=False)
    
    logging.info(f'Predictions saved to {pred_filename}')
    return pred_filename


def predict():
    logging.basicConfig(level=logging.INFO)
    
    try:
        model = load_latest_model()
        
        test_data = load_test_data()
        
        predictions = make_predictions(model, test_data)
        
        save_predictions(predictions)
        
        logging.info('Prediction completed successfully')
        return True
    
    except Exception as e:
        logging.error(f'Error during prediction: {str(e)}')
        raise e


if __name__ == '__main__':
    predict()