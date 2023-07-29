import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_path = "./data/"
data = pd.read_csv(data_path + "data_preprocessed.csv")

# Problematic events summary
fpflag_nt_summary = data.groupby('koi_fpflag_nt').size()
fpflag_co_summary = data.groupby('koi_fpflag_co').size()
fpflag_ec_summary = data.groupby('koi_fpflag_ec').size()
fpflag_ss_summary = data.groupby('koi_fpflag_ss').size()

print("koi_fpflag_nt summary:")
print(fpflag_nt_summary)

print("koi_fpflag_co summary:")
print(fpflag_co_summary)

print("koi_fpflag_ec summary:")
print(fpflag_ec_summary)

print("koi_fpflag_ss summary:")
print(fpflag_ss_summary)

# Temperature analysis
plt.figure(figsize=(10, 5))
sns.scatterplot(data=data, x=data.index, y='koi_teq', hue='target', palette='Set1', s=5)
plt.show()

teq_summary = data.groupby('target')['koi_teq'].agg(['mean', 'median'])
print("koi_teq summary:")
print(teq_summary)

# Stellar effective temperature analysis
plt.figure(figsize=(10, 5))
sns.scatterplot(data=data, x=data.index, y='koi_steff', hue='target', palette='Set1', s=5)
plt.show()

steff_summary = data.groupby('target')['koi_steff'].agg(['mean', 'median'])
print("koi_steff summary:")
print(steff_summary)

# Stars magnitude analysis
plt.figure(figsize=(10, 5))
sns.scatterplot(data=data, x=data.index, y='koi_kepmag', hue='target', palette='Set1', s=5)
plt.show()

kepmag_summary = data.groupby('target')['koi_kepmag'].agg(['mean', 'median'])
print("koi_kepmag summary:")
print(kepmag_summary)

# Planetary radius analysis
plt.figure(figsize=(10, 5))
sns.scatterplot(data=data, x=data.index, y='koi_prad', hue='target', palette='Set1', s=5)
plt.show()

prad_summary = data.groupby('target')['koi_prad'].agg(['mean', 'median'])
print("koi_prad summary:")
print(prad_summary)

# Celestial sphere visualization
plt.figure(figsize=(10, 5))
sns.scatterplot(data=data, x='ra', y='dec', hue='target', palette='Set1', s=5)
plt.show()

# Target variable analysis
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x='target', kde=False)
plt.show()

target_summary = data['target'].value_counts()
print("target variable summary:")
print(target_summary)
