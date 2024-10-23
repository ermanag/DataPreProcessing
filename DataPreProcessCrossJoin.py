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

# 2. Adım: Çapraz Birleştirme (Cross Join)
# Sabit bir sütun ekleyerek her satırın diğer veri setindeki tüm satırlarla eşleşmesini sağlıyoruz
df1_imputed['key'] = 1
df2_imputed['key'] = 1

cross_merged_df = pd.merge(df1_imputed, df2_imputed, on='key').drop('key', axis=1)

print("\nÇapraz Birleştirme (Cross Join) sonrası veri seti:")
print(cross_merged_df)

# 3. Adım: Normalizasyon (Birleştirilmiş veri seti üzerinden)
# a) Min-Max normalizasyon işlemi uyguluyoruz
scaler = MinMaxScaler()
normalized_cross = pd.DataFrame(scaler.fit_transform(cross_merged_df.drop('id_x', axis=1)), columns=cross_merged_df.columns[1:])

print("\nÇapraz Birleştirme sonrası normalizasyon:")
print(normalized_cross)

# b) Z-Score Standardizasyonu
scaler_standard = StandardScaler()
zscore_normalized = pd.DataFrame(scaler_standard.fit_transform(cross_merged_df.drop('id_x', axis=1)), columns=cross_merged_df.columns[1:])
print("\nZ-Score Standardizasyon:")
print(zscore_normalized)

# c) L2 Normu ile Normalizasyon
normalizer = Normalizer(norm='l2')
l2_normalized = pd.DataFrame(normalizer.fit_transform(cross_merged_df.drop('id_x', axis=1)), columns=cross_merged_df.columns[1:])
print("\nL2 Normu ile Normalizasyon:")
print(l2_normalized)
