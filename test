# Data organisation and testing of generalisation ability

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from google.colab import drive

# Google Drive'ı bağlayın
drive.mount('/content/drive')

# CSV dosyasının yolunu belirtin
file_path = '/content/drive/My Drive/valle.csv'

# 1. Veriyi Yükleme ve Ön İşleme

# Veriyi yükleyin
df = pd.read_csv(file_path)

# Özellikler (X) ve hedef değişken (y) ayırma
X = df[['State', 'State Index', 'Selected Action', 'Strategy Used', 'Reward', 'Reward Points']]
y = df['Q-value']

# Kategorik değişkenleri sayısal değerlere dönüştürün (örneğin, One-Hot Encoding ile)
X = pd.get_dummies(X)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme (standartlaştırma)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Model Tanımlama ve Öğrenme Eğrisi Hesaplama

# Model tanımlama (Regresyon modeli: Linear Regression)
model = LinearRegression()

# Öğrenme eğrisini hesaplama
train_sizes, train_scores, test_scores = learning_curve(
    model, 
    X_train, 
    y_train, 
    cv=5, 
    scoring='neg_mean_squared_error', 
    n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Ortalama ve standart sapmaları hesaplayın
train_mean = -train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = -test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

# 3. Öğrenme Eğrisini Görselleştirme

plt.figure()
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Eğitim Skoru (MSE)')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Test Skoru (MSE)')

# Hata aralıklarını çizme
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

plt.xlabel('Eğitim Veri Miktarı')
plt.ylabel('Ortalama Kare Hata (MSE)')
plt.title('Öğrenme Eğrisi')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# 4. Test Setinde Model Performansı

# Modeli eğitim verisi üzerinde eğitin
model.fit(X_train, y_train)

# Test setinde tahmin yapın
y_pred = model.predict(X_test)

# Performans metriklerini hesaplayın
mse = mean_squared_error(y_test, y_pred)
print(f'Test Seti Ortalama Kare Hata (MSE): {mse}')
