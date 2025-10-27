import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        Predict using the saved model and preprocessor.
        """
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Transform data and predict
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        """
        Converts input data into a DataFrame with correct column names for the model.
        """
        try:
            custom_data_input = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],                     # exact match
                "parental level of education": [self.parental_level_of_education],  # exact match
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],   # exact match
                "reading score": [self.reading_score],                       # exact match
                "writing score": [self.writing_score],                       # exact match
            }
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomException(e, sys)
