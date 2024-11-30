import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path ='artifacts\model1.pk1'
            preprocessor_path = 'artifact\preprocessor.pk1'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self, salt1:int, salt2:int, year:int, f1hs:int, f2ls:int):
        self.salt1 = salt1
        self.salt2 = salt2
        self.year = year
        self.f1hs = f1hs
        self.f2ls = f2ls

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "salt1" :[self.salt1],
                "salt2" :[self.salt2],
                "year" :[self.year],
                "f1hs" : [self.f1hs],
                "f2ls" : [self.f2ls],
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)