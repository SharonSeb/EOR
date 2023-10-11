import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pk1")


class DataTransformation:
    def __init__(self):
        super().__init__()
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            num_columns =["$SALT1", "$SALT2", "$YEAR", "$F1HS", "$F1LS"]
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info("Numerical Columns scaling completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(sys, e)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data completed")

            preprocessing_obj = self.get_transformer_object()
            logging.info("Transformer object received")

            trget_column_name = "Objective function_FOE"
            num_columns =["$SALT1", "$SALT2", "$YEAR", "$F1HS", "$F1LS"]

            input_feature_train_df = train_df.drop(columns=[trget_column_name], axis=1)
            target_feature_train_df = train_df[trget_column_name]

            input_feature_test_df = test_df.drop(columns=[trget_column_name], axis=1)
            target_feature_test_df = test_df[trget_column_name]

            logging.info("Applying processing object on training and testing data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved the preprocessing object")


            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessing_obj)
            
            return(train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
            

        except Exception as e:
            raise CustomException(e, sys)

            
            
