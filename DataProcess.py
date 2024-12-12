import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

# Veritabanı bağlantısı
def create_engine_connection(db_name, user, password, host="localhost", port=5432):
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")

# Kaydetmek için veritabanı tanımlayın
source_db_engine = create_engine_connection("internshala_jobs_db", "postgres", "123456")
#
# # CSV'den veriyi oku
data = pd.read_csv("train.csv")
#
# Veriyi kaynak veritabanına kaydet
data.to_sql("train", source_db_engine, if_exists="replace", index=False)
print("Veri kaynak veritabanına kaydedildi.")

# Veriyi kaynak veritabanından al
raw_data = pd.read_sql("SELECT * FROM train", con=source_db_engine)

## Örnek dönüştürme işlemi: "Fare" sütununu normalize etme

#scaler = RobustScaler()
#scaler = MaxAbsScaler()
column_to_normalize = "Fare"
if column_to_normalize in raw_data.columns:
    #raw_data["Fare"] = scaler.fit_transform(raw_data[["Fare"]])
    #raw_data["Fare"] = raw_data["Fare"].apply(lambda x: np.log(x + 1) if x > 0 else 0)
    raw_data[column_to_normalize] = (raw_data[column_to_normalize] - raw_data[column_to_normalize].mean()) / raw_data[column_to_normalize].std()
    # raw_data[column_to_normalize] = (raw_data[column_to_normalize] - raw_data[column_to_normalize].min()) / (
    #             raw_data[column_to_normalize].max() - raw_data[column_to_normalize].min())

# Gürültülü veri örnekleri: Eksik değerler veya beklenmeyen değerler
print("Gürültülü veri tespit ediliyor...")

# Eksik verileri doldurma veya silme
for column in raw_data.columns:
    if raw_data[column].isnull().sum() > 0:
        if raw_data[column].dtype in ["float64", "int64"]:
            raw_data[column].fillna(raw_data[column].mean(), inplace=True)
        else:
            raw_data[column].fillna("Unknown", inplace=True)

# Gürültülü verilerin temizlenmesi: Örnek olarak "Age" değerlerini kontrol etme
if "Age" in raw_data.columns:
    raw_data["Age"] = raw_data["Age"].apply(lambda x: raw_data["Age"].mean() if pd.isnull(x) else x)

# 4. Özellik Seçimi
# Çalışılacak özellikleri seçme (Örnek: Belirli sütunların seçimi)
selected_features = [col for col in raw_data.columns if col in ["Pclass", "Age", "Fare", "Survived"]]
processed_data = raw_data[selected_features]
print("Özellik seçimi tamamlandı.")
print(processed_data.head())

# # Özellikleri ve hedef sütunu ayırma
# X = raw_data.drop(columns=["Survived"])  # Girdi özellikleri
# y = raw_data["Survived"]  # Hedef sütun
#
# # En iyi 3 özelliği seçmek için SelectKBest kullanımı
# selector = SelectKBest(score_func=f_classif, k=3)
# X_new = selector.fit_transform(X, y)
#
# # Seçilen özelliklerin isimleri
# selected_features = X.columns[selector.get_support()]
# processed_data = raw_data[selected_features.tolist() + ["Survived"]]
# print("Özellik seçimi tamamlandı.")
# print(processed_data.head())

# 5. Örneklem Seçimi
# Daha küçük veri seti oluşturma (Örnek: %10 rastgele örnekleme)
sample_data = processed_data.sample(frac=0.1, random_state=42)
print("Örneklem seçimi tamamlandı.")

# İşlemler tamamlandıktan sonra sonuçları kontrol etme
print("Sonuçlardan örnek:")
print(sample_data.head())

# Örneklem sonuçlarını yeni tabloya kaydet
sample_data.to_sql("titanic_result", source_db_engine, if_exists="replace", index=False)
print("Sonuçlar yeni veritabanına kaydedildi.")

