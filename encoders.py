import numpy as np 
import pandas as pd
from sentence_transformers import SentenceTransformer

JOB_FEATURES = ["usr", "jnam", "cnumr", "nnumr", "CR-STR-jobenv-req"]

def sb_encoding_function(df):
    
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    x_str = np.zeros((len(df), 384))
    
    convert_to_str = lambda job: ",".join([f"{job[k]}" for k in job.index if (job[k] and not(pd.isna(job[k])))])
    
    encoded_str = encoder.encode(df[JOB_FEATURES].apply(convert_to_str, axis = 1).values)
    
    for i in range(len(df)):
        x_str[i] = encoded_str[i]
    
    return x_str

def int_encoding_function(df):
    
    JOB_INT_FEATURES = JOB_FEATURES.copy()
    
    for feat in ["usr", "grp", "jtyp", "jnam", "hostname", "CR-STR-jobenv-req"]:
        if feat not in JOB_FEATURES:
            continue
        JOB_INT_FEATURES.remove(feat)
        cat_feat = f"{feat}_cat"
        df.loc[:, cat_feat] = pd.Categorical(df.loc[:, feat]).codes
        JOB_INT_FEATURES.append(cat_feat)
    
    return df[JOB_INT_FEATURES].values