import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

# Örnek veri setleri
data1 = {'id': [1, 2, 3, 4, 5],
         'feature1': [5.1, 6.2, np.nan, 4.5, 6.5],
         'feature2': [2.1, 4.5, 6.7, 8.5, np.nan]}

df1 = pd.DataFrame(data1)

data2 = {'id': [3, 4, 5, 6],
         'feature3': [7.8, 8.9, 5.6, 9.1],
         'feature4': [np.nan, 4.4, 2.5, 7.8]}

df2 = pd.DataFrame(data2)

# 1. Adım: KNN ile Veri Temizleme
imputer = KNNImputer(n_neighbors=2)
df1_imputed = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)
df2_imputed = pd.DataFrame(imputer.fit_transform(df2), columns=df2.columns)

print("KNN ile tamamlanmış veri seti 1:")
print(df1_imputed)
print("\nKNN ile tamamlanmış veri seti 2:")
print(df2_imputed)

# 2. . Adım: Eklemleme (Concatenation / Union)
# İki veri setini satır bazında birleştiriyoruz (Union)
concat_df = pd.concat([df1_imputed, df2_imputed], axis=0, ignore_index=True)

print("\nEklemleme (Concatenation / Union) sonrası veri seti:")
print(concat_df)

# 3. Adım: Normalizasyon (Eklenmiş veri seti üzerinden)
# a) Min-Max normalizasyon işlemi uyguluyoruz
scaler = MinMaxScaler()
normalized_concat = pd.DataFrame(scaler.fit_transform(concat_df.drop('id', axis=1)), columns=concat_df.columns[1:])

print("\nEklemleme sonrası normalizasyon:")
print(normalized_concat)
