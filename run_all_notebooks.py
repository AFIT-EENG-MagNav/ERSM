import sys
import json
import subprocess
import os
from os.path import join

def run_notebook(notebook_file, output_file, **arguments):
    """Pass arguments to a Jupyter notebook, run it and convert to html."""
    '''
    Source: https://felixchenier.uqam.ca/a-better-way-to-run-a-jupyter-notebook-with-arguments/ 
    '''
    
    # Create the arguments file
    with open('arguments.json', 'w') as fid:
        json.dump(arguments, fid)
    # Run the notebook
    subprocess.call([
        'jupyter-nbconvert',
        '--execute',
        '--to', 'pdf',
        '--output', output_file,
        notebook_file])

#---------------------Modify The Following Parameters For Testing--------------------------------------

# IAGA CODES
ERS_IAGA = ['BOU'] # From Data/Extended Reference Station
LRS_IAGA = ['FRD'] # To Data/Local Reference Stations (An Array of Points to Predict At)

# Run the notebook with different arguments
path = 'Test_move_folder' # Name of this folder -- change this for eahc folder you use
print(path)
csv_name = 'testing_output_file.csv'
csv_file = join(path,csv_name)

# Number of Training and Test Days involved in the data set
train = 10
test = 7

#---------------------Don't Modify Below---------------------------------------------------------------

# Change Working Directory to Be In folder above
os.chdir(join(os.getcwd(), '..'))
obj = os.scandir(path)
directory_names = []

for entry in obj :
    if entry.is_dir():
        directory_names.append(entry.name)

directory_names.sort()


# Write Out Header for CSV File 
out_file = open(csv_file, mode = 'a', encoding='UTF8')
header_list = ['ERS', 'LRS', 'Tra_start', 'Tra_end', 'offset', 'scale', 'k', 'w', 'Tra_RMSE_Lin', 'Tra_RMSE_KNN']
tra_day_list = []

#Add Linear Days 
for i in range(train):
    day = i+1
    day_string = str(day)
    if (day<10):
        day_string = '0' + str(day)
    label = 'Tra_Day_Lin_' + day_string
    tra_day_list.append(label)

#Add KNN Days
for i in range(train):
    day = i+1
    day_string = str(day)
    if (day<10):
        day_string = '0' + str(day)
    label = 'Tra_Day_KNN_' + day_string
    tra_day_list.append(label)

# ADD KP-Noise/Quiet Values
tra_day_list.append('Tra_Kp_Quiet_lin')
tra_day_list.append('Tra_Kp_Noise_lin')
tra_day_list.append('Tra_Kp_Quiet_KNN')
tra_day_list.append('Tra_Kp_Noise_KNN')

# Add to header list:
header_list.extend(tra_day_list) 

# Add Testing Day Information 
tst_day_list = ['Tst_start','Tst_end','Tst_RMSE_Lin', 'Tst_RMSE_KNN']

#Add Linear Days 
for i in range(test):
    day = i+1
    day_string = str(day)
    if (day<10):
        day_string = '0' + str(day)
    label = 'Tst_Day_Lin_' + day_string
    tst_day_list.append(label)

#Add KNN Days
for i in range(test):
    day = i+1
    day_string = str(day)
    if (day<10):
        day_string = '0' + str(day)
    label = 'Tst_Day_KNN_' + day_string
    tst_day_list.append(label)

# ADD KP-Noise/Quiet Values
tst_day_list.append('Tst_Kp_Quiet_lin')
tst_day_list.append('Tst_Kp_Noise_lin')
tst_day_list.append('Tst_Kp_Quiet_KNN')
tst_day_list.append('Tst_Kp_Noise_KNN')

# Add To Header List
header_list.extend(tst_day_list) 

# Append To File
out_file.write(','.join(map(str,header_list)))
out_file.write('\n')
out_file.close()

for directory in directory_names :
    
    for extend_loc in ERS_IAGA:
        for local_loc in LRS_IAGA:
            method1 = 'notebook_pdf/'+ directory + '_from_' + extend_loc + '_to_' +local_loc + '_' 'knn_linear.pdf'
            run_notebook('KNN_Final_Clean.ipynb', join(path, method1), data_dir=join(path,directory), csv_file=csv_file, from_data = extend_loc, to_data = local_loc)
            print(method1 + ' done\n')

print('Done with all')


