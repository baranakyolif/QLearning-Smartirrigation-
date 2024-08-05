import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from google.colab import drive

# Google Drive'ı bağlayın
drive.mount('/content/drive')

# CSV dosyasının yolunu belirtin
file_path = '/content/drive/My Drive/alla.csv'

# CSV dosyasını yükleme
df = pd.read_csv(file_path)

# Kategorik sütunları sayısal değerlere dönüştürmek için etiket kodlama
categorical_columns = ['State', 'Selected Action', 'Strategy Used']
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

print("Etiket Kodlama Tamamlandı:")
print(df.head())

# Ölçeklendirme için sayısal sütunları seçme
numeric_columns = df.columns.difference(['Episode', 'Reward', 'Reward Points', 'Q-value'])

# Özellikleri ölçeklendirme
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[numeric_columns])

# GMM modeli oluşturma ve eğitme
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(scaled_features)

# Küme tahminleri ve aykırı değer tespiti
probs = gmm.predict_proba(scaled_features)
threshold = 0.05
outliers = df.index[probs.max(axis=1) < threshold]

print("Aykırı Değerlerin İndeks Numaraları:")
print(list(outliers))

# Görselleştirme

# 1. GMM'in kümelere ayırdığı verileri görselleştirme
# GMM kümelerine göre renklerle işaretleme
gmm_labels = gmm.predict(scaled_features)
df['Cluster'] = gmm_labels

plt.figure(figsize=(12, 6))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=df['Cluster'], palette='viridis', marker='o')
plt.title('GMM Kümeleme Sonuçları')
plt.xlabel('Özellik 1 (ölçeklenmiş)')
plt.ylabel('Özellik 2 (ölçeklenmiş)')
plt.legend(title='Küme')
plt.show()

# 2. Aykırı değerlerin görselleştirilmesi
plt.figure(figsize=(12, 6))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=df.index.isin(outliers), palette={True: 'red', False: 'blue'}, marker='o')
plt.title('Aykırı Değerler')
plt.xlabel('Özellik 1 (ölçeklenmiş)')
plt.ylabel('Özellik 2 (ölçeklenmiş)')
plt.legend(title='Aykırı Değer', loc='best', labels=['Normal', 'Aykırı'])
plt.show()
