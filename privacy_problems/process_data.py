import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os

# convert string to integer
def atoi(s):
    n = 0
    for i in s:
        n = n*10 + ord(i) - ord("0")
    return n

# making folders
outer_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('data',outer_name), exist_ok=True)

# to keep count of each category
angry = 0
fearful = 0
happy = 0
sad = 0
surprised = 0
neutral = 0

df = pd.read_csv('./fer2013.csv')
mat = np.zeros((48,48),dtype=np.uint8)
print("Saving images...")

# read the csv file line by line
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    
    # the image size is 48x48
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

    if df['emotion'][i] == 0 or df['emotion'][i] == 1:
        img.save('data/angry/img'+str(angry)+'.png')
        angry += 1
    elif df['emotion'][i] == 2:
        img.save('data/fearful/img'+str(fearful)+'.png')
        fearful += 1
    elif df['emotion'][i] == 3:
        img.save('data/happy/img'+str(happy)+'.png')
        happy += 1
    elif df['emotion'][i] == 4:
        img.save('data/sad/img'+str(sad)+'.png')
        sad += 1
    elif df['emotion'][i] == 5:
        img.save('data/surprised/img'+str(surprised)+'.png')
        surprised += 1
    elif df['emotion'][i] == 6:
        img.save('data/neutral/img'+str(neutral)+'.png')
        neutral += 1


print("Done!")