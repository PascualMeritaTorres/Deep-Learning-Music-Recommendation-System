# A Novel Content-based Music Recommendation System using Deep Learning Methodologies

## How to reproduce the research project
**1)**: Clone the repository:
```sh
git clone https://github.com/PascualMeritaTorres/Deep-Learning-Music-Recommendation-System.git
```

**2)**: Create a conda environment and install all the required packages
```
conda create -n YOUR_ENV_NAME python=3.7
conda activate YOUR_ENV_NAME
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
├── MachineLearningModelScripts
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
