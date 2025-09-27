import os 
import shutil

#-----------------Modify These Parameters---------------------
train_day = 10
test_day = 7

#----------------Maybe Modify These Parameters----------------
train_data_folder = 'test'
test_data_folder = 'diurnal_test_data'
data_set_name = 'Data_Group'

#-----------------Don't Modify Below--------------------------


# Put Files into convenient folders
obj = os.scandir()
file_names = []
all_files = []
all_folders = set()
for entry in obj :
    if entry.is_file() and (entry.name.endswith('.min') or entry.name.endswith('.sec')):
        file_names.append(entry.name)
        all_files.append(entry)
    elif entry.is_dir():
        all_folders.add(entry.name)     
endings = set()
for name in file_names:
    endings.add(name[3:])


for date in endings: 
    if not (date[:len(date)-8] in all_folders):
        print(date[:len(date)-8])
        os.mkdir(date[:len(date)-8])
    for files in all_files:
        if files.name.lower().endswith(date):
            shutil.move(files.name, date[:len(date)-8])
            
#Organize Folders Into Data Groups
obj2 = os.scandir()
folder_names = []

for entry in obj2:
    if entry.is_dir():
        name = entry.name
        # Some Extra Validation to limit folders that can be looked at
        if name.isdigit() and len(name) == 8:
            folder_names.append(name)
        
sorted_names = sorted(folder_names)
print(sorted_names)
num_cycles = int(len(sorted_names)/(train_day+test_day))
data_set = 1
while data_set <= num_cycles:
    # Get Train List then remove from folder names
    train_list = sorted_names[:train_day]
    del sorted_names[:train_day]
    #print(train_list)
    
    
    # Get Test List the remove from folder names
    test_list = sorted_names[:test_day]
    del sorted_names[:test_day]
    #print(test_list)
    #print(sorted_names)
    
    
    #Training Data Folder
    os.mkdir(train_data_folder)
    
    #Testing Data Folder
    os.mkdir(test_data_folder)
    
    # Move Train Data To Folder
    for day in train_list:
        shutil.move(day, train_data_folder)
        
    # Move Test Data To Folder
    for day in test_list:
        shutil.move(day, test_data_folder)
        
    # Move Train Data Folder and Test Data Folder to new Folder
    data_set_str_num = str(data_set)
    print(data_set)
    if(data_set<10):
    	data_set_str_num = '0'+str(data_set)
    
    print(data_set_name , '_' , data_set_str_num)
    new_folder_name = data_set_name + '_' + data_set_str_num
    os.mkdir(new_folder_name)
    
    shutil.move(train_data_folder, new_folder_name)
    shutil.move(test_data_folder, new_folder_name)
    data_set += 1
    
# Make Notebook Pdf directory
os.mkdir('notebook_pdf')    

