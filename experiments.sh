#!/bin/bash

source activate envp39

cd year_67s
python run_all_notebooks.py
python run_all_neural_networks.py 
cd ..

cd year_34s
python run_all_notebooks.py
python run_all_neural_networks.py 
cd ..

cd year_54n
python run_all_notebooks.py
python run_all_neural_networks.py 
cd ..


