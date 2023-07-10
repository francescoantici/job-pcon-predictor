import numpy as np 
import pandas as pd
from sentence_transformers import SentenceTransformer

JOB_FEATURES = ["usr", "jnam", "cnumr", "nnumr", "jobenv_req"]

def sb_encoding_function(jobs_df:pd.DataFrame) -> np.array:
    
    # Pull the SBert model
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    x_str = np.zeros((len(jobs_df), 384))
    
    # Convert the job data to a comma divided string
    convert_to_str = lambda job: ",".join([f"{job[k]}" for k in job.index if (job[k] and not(pd.isna(job[k])))])
    
    # Encode the jobs
    encoded_str = encoder.encode(jobs_df[JOB_FEATURES].apply(convert_to_str, axis = 1).values)
    
    for i in range(len(jobs_df)):
        x_str[i] = encoded_str[i]
    
    return x_str

def int_encoding_function(jobs_df: pd.DataFrame) -> np.array:
    
    # Create a different set of features' name
    JOB_INT_FEATURES = JOB_FEATURES.copy()
    
    # Encode to categorical the textual features
    for feat in ["usr", "jnam", "hostname", "jobenv_req"]:
        JOB_INT_FEATURES.remove(feat)
        cat_feat = f"{feat}_cat"
        jobs_df.loc[:, cat_feat] = pd.Categorical(jobs_df.loc[:, feat]).codes
        JOB_INT_FEATURES.append(cat_feat)
    
    return jobs_df[JOB_INT_FEATURES].values