# A Novel Content-based Music Recommendation System using Deep Learning Methodologies

## How to reproduce the research project
**1)**: Clone the repository:
```sh
git clone https://github.com/PascualMeritaTorres/Deep-Learning-Music-Recommendation-System.git
```

**2)**: This project can be subdivided into 2 parts, namely data preprocessing which is done inside the Data-Creation-And-Preprocessing, and the training of the machine learning model, which is done inside the Machine-Learning-Models folder. Therefore, to facilitate package versions you must create 2 different environments, for executing commands inside each of the folders. Create a conda environment and install all the required packages for the machine learning model training:

```
cd Machine-Learning-Models
conda env create -f environment.yml -n YOUR_ENV_NAME
```
Create a pip virtual environment and install all the packages for data preprocessing:

```
cd Dataset-Creation-And-Preprocessing
virtualenv YOUR_ENV_NAME
source YOUR_ENV_NAME/bin/activate
pip install -r requirements.txt
```

**3)**: Retrieve spotify data, and preprocess data (See README file under the SpotifyDataPreprocessingScripts folder)
**4)**: Train the model or receive music recommendations from an input song (See README file under the MachineLearningModelScripts folder)

## Repo Structure
```
│
├── Dataset-Creation-And-Preprocessing    <- Serialized Jupyter notebooks created in the project.
│   ├── notebooks                         <- The necessary notebooks to extract and modify Spotify data
│   ├── our_data                          <- Where the dataset will be stored
│   └── README.md                         <- Detailed Instructions to prepare the Spotify data
│
│
├── Machine-Learning-Models
│   ├── models                            <- Stores the pre-trained machine learning models
│   ├── preprocessing                     <- Scripts to preprocess data
│   ├── split                             <- Includes the data split used 
│   ├── test_songs                        <- Dummy-songs used for testing the models
│   ├── training                          <- Scripts to train the model
│   └── README.md                         <- Detailed instructions to train the model or receive music recommendations
|
├── README.md                             <- The document you are currently reading, written for developers to replicate 
|                                         the environment used in the research project
|
└── requirements.txt                      <- The packages that must be installed
```
