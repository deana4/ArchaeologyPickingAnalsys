# Archeology Project

## Overview
This code utilizes the YOLO (You Only Look Once) algorithm to perform object detection on videos. It extracts specific points of objects within the frames, such as tray, brush, and tweezers. The points are then saved in a CSV file for further analysis. The code supports both horizontal and vertical objects, and it can process multiple videos in a given directory.

## Requirements
- Python version: >= 3.10
- OpenCV (`cv2`) library
- tqdm library
- ultralytics library

## Usage
To run the code, execute the following command:

```sh
python <path_to_script> --vid <video_folder_path> --yolo <yolo_folder_path>
```

Alternatively, you can configure the paths in a `config.ini` file and omit the command-line arguments.

### Arguments
- `--vid`: Specifies the path to the video folder.
- `--yolo`: Specifies the path to the YOLO folder.

### Configuration File
You can provide the paths through a configuration file named `config.ini`. The file should be formatted as follows:

```sh
video_path = <video_folder_path>
yolo_path = <yolo_folder_path>
```


Make sure to replace `<video_folder_path>` and `<yolo_folder_path>` with the actual paths on your system.

## Output
The code creates a new folder for each processed video, appending "_YOLO" to its name. Inside this folder, the code generates a CSV file named `<video_name>.csv`. The CSV file contains the following columns:
- `frameID`: Frame number in the video.
- `trayPoints`: Four points representing the tray object (upper-left, upper-right, lower-left, and lower-right). Returns None if the object is horizontal.
- `brushPoint`: The lower-left or lower-right point of the brush object, depending on the specified side ('L' or 'R').
- `TwizzersPoint`: The lower-left or lower-right point of the tweezers object, depending on the specified side ('L' or 'R').

## Important Notes
- The code assumes that the YOLO model files (`best.pt`) for tray, brush, and tweezers are stored in separate subfolders within the YOLO folder.
- The script processes all video files (with extensions `.mp4` or `.mov`) found within the video folder and its subdirectories.
- If a CSV file with the same name already exists in the output path, the script will terminate to avoid overwriting the existing file.
- The code utilizes the YOLO model from the ultralytics library. Make sure you have the required model files in the specified YOLO folder.

Please ensure that you have the necessary dependencies installed and fulfill the requirements mentioned above before running the code.

**If you have any further questions or need assistance, please let me know.**