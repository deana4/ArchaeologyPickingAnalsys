import os
label = '1'
folder_path = 'data/labels'
folder_path_out = 'data/labels_only_class_1'
# loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(folder_path, filename)

        # open the file and read each line
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # create a new file to store the filtered lines
        new_filepath = os.path.join(folder_path_out, filename)
        with open(new_filepath, 'w') as f:
            # loop through each line and check if it starts with '1'
            for line in lines:
                if line.startswith(label):
                    new_line = '0' + line[1:]  # replace first number with 0
                    f.write(new_line)
