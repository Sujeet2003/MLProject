import sys, os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer(self):
        '''
            This function is responsible to transform the data into Numerical Features and Scale the numerical features to a common scale
        '''
        try:
            num_features = ['writing_score', 'reading_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False),)
            ])
            logging.info("Scaling Done for Numerical Features")

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("ohe", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])
            logging.info("Encoding Done for Categorical Features")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_features),
                ("cat_pipeline", cat_pipeline, cat_features),    
            ])
            logging.info("Preprocessing Done over the Training Data")

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
            This function invoke the other function to do their task over datasets.
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test datasets are loaded for the transformation purpose")

            preprocessor_obj = self.get_data_transformer()

            train_target_features = train_df['math_score']
            train_input_features = train_df.drop(columns=['math_score'], axis=1)

            test_target_features = test_df['math_score']
            test_input_features = test_df.drop(columns=['math_score'], axis=1)

            logging.info("Applying Preprocessing Over the datasets...")
            train_transformed_data = preprocessor_obj.fit_transform(train_input_features)
            test_transformed_data = preprocessor_obj.transform(test_input_features)

            train = np.column_stack((train_transformed_data, np.array(train_target_features)))
            test = np.column_stack((test_transformed_data, np.array(test_target_features)))

            logging.info("Preprocessing Done over the Train and Test datasets.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_file_path,
                object = preprocessor_obj,
            )

            logging.info("Preprocessor saved successfully as pickle file")

            return train, test, self.data_transformation_config.preprocessor_file_path
        except Exception as e:
            raise CustomException(e, sys)