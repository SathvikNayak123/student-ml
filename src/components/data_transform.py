import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:
            num_col=['reading_score', 'writing_score']
            cat_col=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False)),
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info('numerical and categorical transformation completed')

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_col),
                    ('cat_pipeline',cat_pipeline,cat_col)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train n test data")

            preprocessing_obj=self.get_data_transformer_obj()
            logging.info("obtained preprocessed obj")

            target_col='math_score'
            num_col=['reading_score', 'writing_score']

            train_df_feature=train_df.drop(columns=[target_col],axis=1)
            train_df_target=train_df[target_col]

            test_df_feature=test_df.drop(columns=[target_col],axis=1)
            test_df_target=test_df[target_col]

            logging.info("split data into train and test")

            train_df_preprocess=preprocessing_obj.fit_transform(train_df_feature)
            test_df_preprocess=preprocessing_obj.transform(test_df_feature)

            train_arr=np.c_[train_df_preprocess,np.array(train_df_target)]
            test_arr=np.c_[test_df_preprocess,np.array(test_df_target)]
            logging.info("applied preprocess obj on train n test")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('saved preprocess obj')

            return(
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e,sys)