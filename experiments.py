from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd 

def online_experiment(jobs_df: pd.DataFrame, models_setup_dict:dict = {}) -> None:
    """Function to run an online evalutation of the model

    Args:
        jobs_df (pd.DataFrame): The DataFrame containing the job entries.
        models_setup_dict (dict, optional): The models' setup to use in the experiments. Defaults to {}.
    """
    
    # Training interval to create the training set 
    training_interval = 60
    
    # Create the submission and end day from the time
    jobs_df["day"] = jobs_df.submit_time.apply(lambda sbt: datetime.fromtimestamp(int(sbt)).date())
    jobs_df["end_day"] = jobs_df.end_time.apply(lambda edt: datetime.fromtimestamp(int(edt)).date())
    
    # Find the list of the days in the dataset
    days = jobs_df.day.unique()
    
    # Compute the evalutation for each day
    for i in tqdm(range(len(days[training_interval:]))):
                            
        try:
    
            true = []
                            
            day = days[training_interval + i]
            
            # Create test set by taking the jobs submitted in the specific day    
            test_df = jobs_df[jobs_df.day == day].sort_values("submit_time")
            
            # Create the training set by taking the jobs submitted and ended in the last training_interval days                                  
            train_df = jobs_df[(jobs_df.end_day >= day - timedelta(days=training_interval)) & (jobs_df.end_time < test_df.submit_time.values[0])]
                        
            if (len(train_df) == 0) or (len(test_df) == 0):
                continue
            
            # Compute prediction and evaluation for each model
            for experiment_name in models_setup_dict:
                                
                experiment_setting = models_setup_dict[experiment_name]
                
                encoding_function = experiment_setting["encoding_function"]
                target_feat = experiment_setting["target_feat"]
                model = experiment_setting["model"]
                evaluation_function = experiment_setting["evaluation_function"] 
                            
                x_train = encoding_function(train_df)
                x_test = encoding_function(test_df)
                
                y_train = train_df[target_feat].values
                y_test = test_df[target_feat].values
                                
                model = model(**experiment_setting["hyperparameters"]).fit(x_train, y_train)
                                                
                preds = list(model.predict(x_test))
                
                true = list(y_test)
                                
                print(evaluation_function(true, preds))
                         
        except Exception as e:
            print(e)

def offline_experiment(jobs_df: pd.DataFrame, models_setup_dict:dict = {}) -> None:
    """Function to run an offline evalutation of the models

    Args:
        jobs_df (pd.DataFrame): The DataFrame containing the job entries.
        models_setup_dict (dict, optional): The models' setup to use in the experiments. Defaults to {}.
    """
    
    # Size of the test test 
    test_size = 0.3
    
    # Compute the evaluation for each model
    for experiment_name in tqdm(models_setup_dict):
        
        try:
        
            experiment_setting = models_setup_dict[experiment_name]
            
            encoding_function = experiment_setting["encoding_function"]
            target_feat = experiment_setting["target_feat"]
            model = experiment_setting["model"]
            evaluation_function = experiment_setting["evaluation_function"]
            
            # Find the boundaries for the train and test set accordingly to test_size               
            train_val = int(len(df)*(1 - test_size))
            
            train_df = df.iloc[:train_val]
            test_df = df.iloc[train_val:]
            
            x_train = encoding_function(train_df)
            x_test = encoding_function(test_df)
            
            y_train = train_df[target_feat].values
            y_test = test_df[target_feat].values
                
            model = model(**experiment_setting["hyperparameters"]).fit(x_train, y_train)
            
            print(evaluation_function(y_test, model.predict(x_test)))
            
        
        except Exception as e:
            print(e)
            continue 
    
