# Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

# Örnek veri setleri
# İlk veri seti
data1 = {'id': [1, 2, 3, 4, 5],
         'feature1': [5.1, 6.2, np.nan, 4.5, 6.5],
         'feature2': [2.1, 4.5, 6.7, 8.5, np.nan]}

df1 = pd.DataFrame(data1)

# İkinci veri seti
data2 = {'id': [3, 4, 5, 6],
         'feature3': [7.8, 8.9, 5.6, 9.1],
         'feature4': [np.nan, 4.4, 2.5, 7.8]}

df2 = pd.DataFrame(data2)

# 1. Adım: KNN ile Veri Temizleme
# Kayıp değerleri tamamlamak için KNNImputer kullanıyoruz
imputer = KNNImputer(n_neighbors=2)
df1_imputed = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)
df2_imputed = pd.DataFrame(imputer.fit_transform(df2), columns=df2.columns)

print("KNN ile tamamlanmış veri seti 1:")
print(df1_imputed)
print("\nKNN ile tamamlanmış veri seti 2:")
print(df2_imputed)

# 2. Adım: Veri Birleştirme
# İki veri setini 'id' sütunu üzerinden birleştiriyoruz
merged_df = pd.merge(df1_imputed, df2_imputed, on='id', how='inner')

print("\nBirleştirilmiş veri seti:")
print(merged_df)

# 3. Adım: Normalizasyon
# a) Min-Max normalizasyon işlemi uyguluyoruz
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(merged_df.drop('id', axis=1)), columns=merged_df.columns[1:])

print("\nNormalizasyon sonrası veri seti:")
print(normalized_data)

# b) Z-Score Standardizasyonu
scaler_standard = StandardScaler()
zscore_normalized = pd.DataFrame(scaler_standard.fit_transform(merged_df.drop('id', axis=1)), columns=merged_df.columns[1:])
print("\nZ-Score Standardizasyon:")
print(zscore_normalized)

# c) L2 Normu ile Normalizasyon
normalizer = Normalizer(norm='l2')
l2_normalized = pd.DataFrame(normalizer.fit_transform(merged_df.drop('id', axis=1)), columns=merged_df.columns[1:])
print("\nL2 Normu ile Normalizasyon:")
print(l2_normalized)
