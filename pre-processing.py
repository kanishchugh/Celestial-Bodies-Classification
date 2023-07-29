import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries import
from sklearn.preprocessing import LabelEncoder

# Read the original data
data_original = pd.read_csv("./data/cumulative.csv")
print(data_original.describe())

# Select variables of interest and remove unnecessary columns
columns_to_remove = ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_tce_delivname', 
                     'koi_tce_plnt_num'] + [col for col in data_original.columns if 'err' in col]
data = data_original.drop(columns=columns_to_remove)


# Remove observations with missing values for koi_teq and koi_kepmag
data = data.dropna(subset=['koi_teq', 'koi_kepmag'])

# Analyze koi_score missing values
data_na_score = data[data['koi_score'].isna()]
print("koi_score NA: among candidate =", data_na_score[data_na_score['koi_disposition'] == 'CANDIDATE'].shape[0])
print("koi_score NA: among confirmed =", data_na_score[data_na_score['koi_disposition'] == 'CONFIRMED'].shape[0])
print("koi_score NA: among false positive =", data_na_score[data_na_score['koi_disposition'] == 'FALSE POSITIVE'].shape[0])
print("Total:", data_na_score.shape[0])

# Analyze the differences between koi_disposition and koi_pdisposition
data_grouped = data.groupby(['koi_disposition', 'koi_pdisposition']).size().reset_index(name='tot')
print(data_grouped)

# Remove koi_pdisposition column and drop rows with missing koi_score
data = data.drop(columns=['koi_pdisposition']).dropna(subset=['koi_score'])

# Binarize the target variable
data['target'] = LabelEncoder().fit_transform(data['koi_disposition'] == 'CONFIRMED').astype(int)
data = data.drop(columns=['koi_disposition'])

# Save the preprocessed data
data.to_csv('./data/data_preprocessed.csv', index=False)
