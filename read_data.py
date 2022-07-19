from time import sleep
from tqdm import tqdm
import pandas as pd 
dataframe = pd.read_csv('C:/Users/Dell/Desktop/Trabajo de titulación/Dataset/dataset_final.csv')
short = 0
medium = 0
long = 0
for idx in tqdm(range(len(dataframe))):
    if(len(dataframe['texto'].iloc[idx])>1300 and len(dataframe['texto'].iloc[idx])<2000):
        short = short+1
        # print(dataframe['texto'].iloc[idx])
    elif(len(dataframe['texto'].iloc[idx])>=2000 and len(dataframe['texto'].iloc[idx])<3300):
        medium = medium+1
    elif(len(dataframe['texto'].iloc[idx])>=3300):
        long = long+1

    
print(f"Cortas...{short}")
print(f"Medias...{medium}")
print(f"Largas...{long}")