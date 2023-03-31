# Instructions to use the Sptify API data



## Spotify Keys Creation
1. Create a new app in [Spotify for Developer's Dashboard ](https://developer.spotify.com/dashboard)
2. Retrieve the CLIENT_ID and CLIENT_SECRET of your app
3. Change the variables 'client_id' and 'client_secret' in spotify_mp3_download.ipynb to the ones retrieved in step 2
4. Export the path variables. To do so execute the command 
```sh
nano ~/.zshrc
```
and at the end of the document do 
```sh
export SPOTIPY_CLIENT_ID='your_client_id'
export SPOTIPY_CLIENT_SECRET='your_client_secret'
```
5. Execute 
```sh
source ~/.zshrc
```



## Data Preprocessing (All Jupyter Notebooks are located under the notebooks folder)
1. Run the first cell of data_preprocessing.ipynb to remove the irrelevant columns of the initial dataset
2. Run the first cell of spotify_mp3_download.ipynb to download the spotify songs.
3. Run the second cell of data_preprocessing.ipynb to modify the csv to only include downloaded songs (there can be problems with spotify) 
4. Run the data_binarization.ipynb cells in order to binarize some columns
5. Run the check_audio_length.ipynb to trim audio (There may be songs that are longer than 30 seconds, and some that are shorter)
6. Run create_files_from_csv.ipynb


