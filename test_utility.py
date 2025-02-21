import pytest
import pandas as pd
import numpy as np
from prediction_demo import data_preparation,data_split,train_model,eval_model

@pytest.fixture
def housing_data_sample():
    return pd.DataFrame(
      data ={
      'price':[13300000,12250000],
      'area':[7420,8960],
    	'bedrooms':[4,4],	
      'bathrooms':[2,4],	
      'stories':[3,4],	
      'mainroad':["yes","yes"],	
      'guestroom':["no","no"],	
      'basement':["no","no"],	
      'hotwaterheating':["no","no"],	
      'airconditioning':["yes","yes"],	
      'parking':[2,3],
      'prefarea':["yes","no"],	
      'furnishingstatus':["furnished","unfurnished"]}
    )

def test_data_preparation(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    # Target and datapoints has same length
    assert feature_df.shape[0]==len(target_series)

    #Feature only has numerical values
    assert feature_df.shape[1] == feature_df.select_dtypes(include=(np.number,np.bool_)).shape[1]

@pytest.fixture
def feature_target_sample(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    return (feature_df, target_series)

def test_data_split(feature_target_sample):
    return_tuple = data_split(*feature_target_sample)
    # TODO test if the length of return_tuple is 4
    # raise NotImplemented

    # Test if the length of return_tuple is 4
    assert len(return_tuple) == 4, "data_split should return exactly four outputs."
    
    # (Optional) You can add more validation tests below, e.g.:
    X_train, X_test, y_train, y_test = return_tuple
    
    # Check that the train and test data are not empty
    assert len(X_train) > 0, "X_train should not be empty"
    assert len(X_test) > 0, "X_test should not be empty"
    assert len(y_train) > 0, "y_train should not be empty"
    assert len(y_test) > 0, "y_test should not be empty"

    # Check that the sum of lengths equals the original data length
    original_length = len(feature_target_sample[0])
    assert (len(X_train) + len(X_test)) == original_length, \
        "Combined train and test features should match original feature count"
    assert (len(y_train) + len(y_test)) == len(feature_target_sample[1]), \
        "Combined train and test targets should match original target count"