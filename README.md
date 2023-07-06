# Job Power Prediction in HPC systems

This repository contains the source code for the submission of the paper: Natural Language Processing for Online Job Power Prediction in HPC Systems to SHIPS @ SC23.

## Repository structure 

- `encoders.py` contains the code for the job features encoding presented in the paper.
- `evaluate.py` contains the model definition and the code to launch the experiments. 
- `experiments.py` contains the code for the experimental setting presented in the paper, namely offline and online. 
- `utils.py` contains the code for the evaluation of the results with the different metrics considered in the study.
- `requirements.txt` is the list of the python packages required to run the code.  

## Data Structure 

The data to use in order to replicate the experiments should be structured as follows. 

The jobs data should be contained in a dataframe, saved as a parquet file (or a format recognizable by pandas). 

Each job should contain the following features:

- `submit_time` : the time of submission of the job;
- `end_time` : the time of the stop of the job's execution;
- `cnumr` : the number of cores requested by the user for the job;
- `nnumr` : the number of nodes requested by the user for the job;
- `CR-STR-jobenv-req` : the environment requested for the job execution.
- `nunma` : the number of nodes allocated to the job for the execution;
- `maxpcon` : the maximum value of the job power consumption;
- `avgpcon` : the average value of the job power consumption;
- `usr` : the name of the user submitting the job;
- `jnam` : the name of the job;

## Launch the experiments 

In order to replicate the experiments reported in the paper, the `evaluate.py` script must be executed. 

All the packages used in the project are reported in the `requirements.txt`, the python version used was the 3.11.3. 

It would be useful (but not compulsory) to create a virtual environment and then install the required packages with `pip3 install -r requirements.txt`. 

After setting up the python environment, the `evaluate.py` file should be modified inserting the path to the data file (in parquet) in the `job_df_path` variable.

It is possible to set different random seeds for the definition of the models by modifying the `RANDOM_SEED` variable, originally initialized to 42.
