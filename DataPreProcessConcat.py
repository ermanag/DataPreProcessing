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

def knn_imputer(data, k=2):
    # Veri setini bir numpy array'ine çevir
    data_matrix = data.values
    n_samples, n_features = data_matrix.shape

    # Eksik değer olan satırların indekslerini bul
    missing_indices = np.argwhere(np.isnan(data_matrix))

    # Eksik değerleri tamamlamak için her bir eksik hücreyi işle
    for row_idx, col_idx in missing_indices:
        # Geçerli satır ve sütun
        target_row = data_matrix[row_idx]

        # Bu satırla diğer tüm satırlar arasındaki mesafeyi hesapla (Eksik değer içermeyen satırlar)
        distances = []
        for i in range(n_samples):
            if i != row_idx and not np.isnan(data_matrix[i, col_idx]):
                distance = 0
                # Öklid mesafesini hesapla (Eksik olmayan sütunlar üzerinde)
                for j in range(n_features):
                    if j != col_idx and not np.isnan(target_row[j]) and not np.isnan(data_matrix[i, j]):
                        distance += (target_row[j] - data_matrix[i, j]) ** 2
                distances.append((i, np.sqrt(distance)))

        # Mesafeye göre komşuları sırala ve en yakın k komşuyu seç
        distances.sort(key=lambda x: x[1])
        nearest_neighbors = [data_matrix[i][col_idx] for i, _ in distances[:k]]

        # Eksik değeri en yakın k komşunun ortalaması ile tamamla
        data_matrix[row_idx, col_idx] = np.mean(nearest_neighbors)

    # İmpute edilmiş veri setini döndür
    return pd.DataFrame(data_matrix, columns=data.columns)

df1_imputed = knn_imputer(df1, k=2)
df2_imputed = knn_imputer(df2, k=2)

# imputer = KNNImputer(n_neighbors=2)
# df1_imputed = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)
# df2_imputed = pd.DataFrame(imputer.fit_transform(df2), columns=df2.columns)

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
