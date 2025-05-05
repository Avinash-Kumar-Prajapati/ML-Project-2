import pickle
import os
import sys

from src.exception import CustomException

def save_object(obj, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
           object=pickle.load(file_obj)
           return object

    except Exception as e:
        raise CustomException(e, sys)
    
def parse_memory(x):
    try:
        SSD=HDD=Flash=Hybrid=0
        parts=x.split('+')
        for part in parts:
            memory=part.strip()
            storage=memory.split()[0]

            if 'TB' in storage:
                storage=int(float(storage.replace('TB', '')))*1000
            else:
                storage=int(storage.replace('GB', ''))

            if 'SSD' in memory:
                SSD=storage
            elif 'HDD' in memory:
                HDD=storage
            elif 'Flash Storage' in memory:
                Flash=storage
            elif 'Hybrid' in memory:
                Hybrid=storage

        return [SSD, HDD, Flash, Hybrid]
    
    except Exception as e:
        raise CustomException(e, sys)
    

def remove_outliers(df):
    try:
        Q1 = df['Price'].quantile(0.25)
        Q3 = df['Price'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]

        return df
    
    except Exception as e:
        raise CustomException(e, sys)
    

def remove_word(text):
                for word in ['Touchscreen', 'IPS Panel', '/']:
                    text = text.replace(word, "")
                return text