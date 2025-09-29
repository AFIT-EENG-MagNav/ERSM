[![DOI](https://zenodo.org/badge/1065409449.svg)](https://doi.org/10.5281/zenodo.17226837)

**Distribution A**: Approved for Public Release. Distribution Unlimited. 

The views expressed herein are those of the authors and do not necessarily reflect the official policy or position of the United States Government, Department of Defense, United States Air Force or Air University. 



This repository contains the notebooks and scripts used for ```ERSM Paper Title Here```. 

### Repository Organization
* ```data_and_results```: Contains the raw data used for the experiments and the raw results (not processed nor visualized). It divided up the results by the final paper, AGU poster presentation, and a test conducted using data collected by Edwards AFB. 
* ```*.ipynb```: These are the notebooks used to run the kNN and neural-net models. All scripts include the linear regression model (as computed in ```MAMMAL```).
* ```mammal```: Contains the version of the [```MAMMAL```](https://github.com/PowerBroker2/MAMMAL) library used in this work. 
* ```*.py```: Utility scripts used to run data and move the raw data into a directory structure usable by the notebooks. 
* ```old_files.zip```: Files that were used in our preliminary testing and development. 
* ```preliminary_experiments.zip```: These are experiments we ran while developing extensions to the model.
* ```experiments.sh```: A sample script to run several experiments synchronously.   
* ```exp_env.yml```: The environment used to run experiments.
