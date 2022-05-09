from pathlib import Path
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
image_dir = Path('C:\CLASS\python\Rec\Indian_food')

filepaths = list(image_dir.glob(r'**\*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([filepaths, labels], axis=1)

category_samples = []
for category in images['Label'].unique():
    category_slice = images.query("Label == @category")
    category_samples.append(category_slice.sample(2000, replace=True, random_state=1))
image_df = pd.concat(category_samples, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)

image_df.to_pickle('C:\CLASS\python\Rec\imagef1.pkl')

print(image_df['Label'].value_counts())

data_df = pd.read_pickle('C:\CLASS\python\Rec\imagef1.pkl')
# data_df

train_df, test_df = train_test_split(data_df, train_size=0.8, shuffle=True, random_state=1)
train_df.to_pickle('C:\CLASS\python\Rec\\ftrain1.pkl')
test_df.to_pickle('C:\CLASS\python\Rec\\ftest1.pkl')
