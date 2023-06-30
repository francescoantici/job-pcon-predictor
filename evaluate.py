import pandas as pd 
from encoders import sb_encoding_function, int_encoding_function
from utils import regression_report
from experiments import online_experiment, offline_experiment
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor

if __name__ == "__main__":
    
    RANDOM_SEED = 42
    
    job_df_path = ""
    
    job_df = pd.read_csv(job_df_path).sort_values("submit_time")
    
    job_df["nmaxpcon"] = job_df.apply(lambda job: int(int(job.maxpcon)/int(job.nnuma))).values 
    job_df["navgpcon"] = job_df.ppr.apply(lambda job: int(int(job.avgpcon)/int(job.nnuma))).values
    
    models_setup_dict = {
        "SB+RF_MAX": 
            {
                "target_feat" : "nmaxpcon",
                "model" : RandomForestRegressor,
                "evaluation_function" : regression_report,
                "encoding_function" : sb_encoding_function, 
                "hyperparameters" : {
                    "n_jobs" : -1,
                    "random_state" : RANDOM_SEED
                }
            },
        "INT+RF_MAX": 
            {
                "target_feat" : "nmaxpcon",
                "model" : RandomForestRegressor,
                "evaluation_function" : regression_report,
                "encoding_function" : int_encoding_function,
                "hyperparameters" : {
                    "n_jobs" : -1,
                    "random_state" : RANDOM_SEED
                }
            },
        "SB+XG_MAX": 
            {
                "target_feat" : "nmaxpcon",
                "model" : XGBRegressor,
                "evaluation_function" : regression_report,
                "encoding_function" : sb_encoding_function, 
                "hyperparameters" : {
                    "n_jobs" : -1,
                    "random_state" : RANDOM_SEED,
                 
                }
                
            },
        "INT+XG_MAX": 
            {
                "target_feat" : "nmaxpcon",
                "model" : XGBRegressor,
                "evaluation_function" : regression_report,
                "encoding_function" : int_encoding_function,
                "hyperparameters" : {
                    "n_jobs" : -1,
                    "random_state" : RANDOM_SEED,
              
                }
            },
        "SB+AD_MAX": 
            {
                "target_feat" : "nmaxpcon",
                "model" : AdaBoostRegressor,
                "evaluation_function" : regression_report,
                "encoding_function" : sb_encoding_function, 
                "hyperparameters" : {
                    "random_state" : RANDOM_SEED
                }
                
            },
        "INT+AD_MAX": 
            {
                "target_feat" : "nmaxpcon",
                "model" : AdaBoostRegressor,
                "evaluation_function" : regression_report,
                "encoding_function" : int_encoding_function,
                "hyperparameters" : {
                    "random_state" : RANDOM_SEED,
                }
            },
        "SB+RF_AVG": 
            {
                "target_feat" : "navgpcon",
                "model" : RandomForestRegressor,
                "evaluation_function" : regression_report,
                "encoding_function" : sb_encoding_function, 
                "hyperparameters" : {
                    "n_jobs" : -1,
                    "random_state" : RANDOM_SEED
                }
            },
        "INT+RF_AVG": 
            {
                "target_feat" : "navgpcon",
                "model" : RandomForestRegressor,
                "evaluation_function" : regression_report,
                "encoding_function" : int_encoding_function,
                "hyperparameters" : {
                    "n_jobs" : -1,
                    "random_state" : RANDOM_SEED
                }
            },
        "SB+XG_AVG": 
            {
                "target_feat" : "navgpcon",
                "model" : XGBRegressor,
                "evaluation_function" : regression_report,
                "encoding_function" : sb_encoding_function, 
                "hyperparameters" : {
                    "n_jobs" : -1,
                    "random_state" : RANDOM_SEED,
                   
                }
                
            },
        "INT+XG_AVG": 
            {
                "target_feat" : "navgpcon",
                "model" : XGBRegressor,
                "evaluation_function" : regression_report,
                "encoding_function" : int_encoding_function,
                "hyperparameters" : {
                    "n_jobs" : -1,
                    "random_state" : RANDOM_SEED,
                    
                }
            },
        "SB+AD_AVG": 
            {
                "target_feat" : "navgpcon",
                "model" : AdaBoostRegressor,
                "evaluation_function" : regression_report,
                "encoding_function" : sb_encoding_function, 
                "hyperparameters" : {
                    "random_state" : RANDOM_SEED
                }
                
            },
        "INT+AD_AVG": 
            {
                "target_feat" : "navgpcon",
                "model" : AdaBoostRegressor,
                "evaluation_function" : regression_report,
                "encoding_function" : int_encoding_function,
                "hyperparameters" : {
                    "random_state" : RANDOM_SEED,
                }
            },
    }
    
    online_experiment(jobs_df = job_df, models_setup_dict = models_setup_dict)
    