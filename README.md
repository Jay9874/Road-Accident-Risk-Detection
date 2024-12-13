# Road Accident Risk Detection

Python script that detect faces on the image or video, extracts them and saves to the specified folder.
Original author [freearhey](https://github.com/freearhey/face-extractor).

After the face has been extracted, the eye is too extracted from the face.
Original author [edge7](https://github.com/edge7/Eye-Region-Extraction-Toolbox).

Using the face, the facial expression is recorded and stored in status.
Original author [infoaryan](https://github.com/infoaryan/Driver-Drowsiness-Detection)

## Prerequisites
- Python 3.9.20

- Download face landmark dataset from [kaggle](https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat) or [GitHub](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) and place it in root directory.

## Installation

Copy repository to your computer using one of the available methods. For example, this can be done using the `git clone` command:

```sh
git clone https://github.com/Shubham-Lal/Road-Accident-Risk-Detection.git
```

Then you need to go to the project folder and install all the dependencies:

```sh
# change directory
cd Road-Accident-Risk-Detection

# install dependencies
pip install -r requirements.txt
```

And you're done.

## Demo

```sh
python main.py
```

## Usage - extract.py

In the `input` folder you can find several images that can be processed using the script, like so:

```sh
python extract.py --input input
```

To run the script you need to pass only the path to the image that need to be processed, as well as the path to the folder where the extracted faces will be saved.

```sh
python extract.py --input path/to/image.jpg --output path/to/output_folder
```

The video file can also be used as the input:

```sh
python extract.py --input path/to/video.mp4 --output path/to/output_folder
```

Or it could be a folder containing these files:

```sh
python extract.py --input path/to/folder_with_images
```

By default, the files are saved in the `output` folder.

**Arguments:**

- `-h, --help`: show this help message and exit
- `-i, --input`: path to input directory or file
- `-o, --output`: path to output directory of faces
- `-p, --padding`: padding ratio around the face (default: 1.0)