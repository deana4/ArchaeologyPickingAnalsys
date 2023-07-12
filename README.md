# Archeology Project

## Overview
- This code analyse frames of archeology picking mission's video and extracting features of the picker by arranging them in a CSV file containing the directory of the videos used. Before running this code, you should extract the YOLO recognition CSV file. To do so, go into Yolo_Extractor directory and follow the instructions.
- The length of the videos is divided by the following rule: a 30 minutes video of picking -> into 2 videos of 5 minutes. first five minutes and last five minutes of the 30 minutes video.

## Requirements
- Python version: >= 3.10
- All files and directories MainTracker requiring for running.

## Usage
you can configure the paths in a `featuresConfig.ini` file and omit the command-line arguments.

### Configuration File
You need to provide the paths through a configuration file named `config.ini`. The file should be formatted as follows:

```sh
video_path = <video_folder_path>
csv_path = <csv_folder_path>
csv_features_path = <full_features_csv_path>
```
To run the code, execute the following command:
```sh
python <path_to_script> 
```

Make sure to replace `<video_folder_path>` and `<yolo_folder_path>` with the actual paths on your system.

## Output

- The code creates a CSV file containing all the features extracted from a directory of videos and csvs.
- CSV format is: countMatBrush, AlreadyVisitedBrushMat,
                 %OfSandBrushes, %OfPaperBrushes,
                 countMatTwizzers, 'AlreadyVisitedTwizzersMat,
                 %OfSandTwizzers, %OfPaperTwizzers, Directory
- This features is extracted twice, once for the first_five_minutes.mp4 and one for the last_five_minutes.mp4 and concat the results into one, for each pair of videos.


## Important Notes
- The code assumes that you have two directories: videos and csvs. in each one, there will be directories containing the first five and last five videos. Accordingly, in the csvs directory there will be directories containing the first five and last five csv files.
- You need to match the directories name inside each folder: videos and csvs. For example: Inside videos, you will have directory named "first duo" that contains 'first_five.mp4' and 'last_five.mp4' minutes. Same way goes for their csv files: Inside csvs, you will have directory named "first duo" that contains 'first_five.csv' and 'last_five.csv'
- The script processes all video files (with extensions `.mp4` or `.mov`) found within the video folder and its subdirectories.
- If a CSV file with the same name already exists in the output path, the script will ask you whether to re-write it or not.

Please ensure that you have the necessary dependencies installed and fulfill the requirements mentioned above before running the code.
