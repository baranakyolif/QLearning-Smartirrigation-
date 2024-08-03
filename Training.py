import numpy as np
import random
import pandas as pd
from google.colab import drive
import os

# Google Drive'ı bağla
drive.mount('/content/drive')

# Eylemler
actions = ["Yüksek Sulama", "Orta Sulama", "Düşük Sulama", "Sulama Yapmamak"]
import random

# Ödül Fonksiyonları
temperature_rewards = {'0°C': -2,'1°C':   ETC. Assign a value to each variable in order of importance in your data study}
humidity_rewards = {  '0%': +3, '1%': +3, '2%': +3 ETC. Assign a value to each variable in order of importance in your data study}
wind_speed_rewards = {'0 km/saat': 0, '1 km/saat': 0,  ETC. Assign a value to each variable in order of importance in your data study
soil_moisture_rewards = { '0%': +4, '1%': +4, '2%': +4,  ETC. Assign a value to each variable in order of importance in your data study}
soil_ph_rewards = {  '3.0': -3, '3.1': -3, ETC. Assign a value to each variable in order of importance in your data study}
plant_age_rewards = {'1 yıl': +1,  ETC. Assign a value to each variable in order of importance in your data study}
soil_nutrient_rewards = {'Yüksek': +1,  ETC. Assign a value to each variable in order of importance in your data study}
soil_type_rewards = {'Killi Toprak': +2,  ETC. Assign a value to each variable in order of importance in your data study}

# Rastgele değerler seçme
def generate_random_conditions():
    temperature = random.choice(list(temperature_rewards.keys()))
    humidity = random.choice(list(humidity_rewards.keys()))
    wind_speed = random.choice(list(wind_speed_rewards.keys()))
    soil_moisture = random.choice(list(soil_moisture_rewards.keys()))
    plant_age = random.choice(list(plant_age_rewards.keys()))
    soil_ph = random.choice(list(soil_ph_rewards.keys()))
    soil_nutrient = random.choice(list(soil_nutrient_rewards.keys()))
    soil_type = random.choice(list(soil_type_rewards.keys()))
    
    conditions = [temperature, humidity, wind_speed, soil_moisture, plant_age, soil_ph, soil_nutrient, soil_type]
    return conditions

# Ödül Fonksiyonu
def reward_function(conditions):
    if not isinstance(conditions, (list, tuple)) or len(conditions) != 8:
        raise ValueError("Conditions must be a list or tuple of length 8.")
    
    temperature, humidity, wind_speed, soil_moisture, plant_age, soil_ph, soil_nutrient, soil_type = conditions
    
    temp_reward = temperature_rewards[temperature]
    humidity_reward = humidity_rewards[humidity]
    wind_speed_reward = wind_speed_rewards[wind_speed]
    soil_moisture_reward = soil_moisture_rewards[soil_moisture]
    plant_age_reward = plant_age_rewards[plant_age]
    soil_ph_reward = soil_ph_rewards[soil_ph]
    soil_nutrient_reward = soil_nutrient_rewards[soil_nutrient]
    soil_type_reward = soil_type_rewards[soil_type]
    
    total_reward = (temp_reward + humidity_reward + wind_speed_reward +
                    soil_moisture_reward + plant_age_reward +
                    soil_ph_reward + soil_nutrient_reward +
                    soil_type_reward)
    
    return total_reward

 

# Eylem doğruluk puanı hesaplama
def reward_points_based_on_action(action, total_reward):
    if action == "Yüksek Sulama":
        if total_reward >= 9:
            return min(max(1, total_reward - 8), 5)  # Ödül puanı aralığı 1-5
        return 0
    elif action == "Orta Sulama":
        if 4 <= total_reward < 9:
            return min(max(1, total_reward - 3), 5)  # Ödül puanı aralığı 1-5
        return 0
    elif action == "Düşük Sulama":
        if -1 <= total_reward < 4:
            return min(max(1, total_reward + 1), 5)  # Ödül puanı aralığı 1-5
        return 0
    elif action == "Sulama Yapmamak":
        if total_reward < -1:
            return min(max(1, -total_reward), 5)  # Ödül puanı aralığı 1-5
        return 0

  # Örnek kullanım
random_conditions = generate_random_conditions()
total_reward = reward_function(random_conditions)
actions = ["Yüksek Sulama", "Orta Sulama", "Düşük Sulama", "Sulama Yapmamak"]

print(f"Conditions: {random_conditions}")
print(f"Total Reward: {total_reward}")

for action in actions:
    points = reward_points_based_on_action(action, total_reward)
    print(f"Action: {action}, Points: {points}")

    
# Softmax Fonksiyonu
def softmax(Q, state_index, temperature):
    exp_values = np.exp(Q[state_index] / temperature)
    probabilities = exp_values / np.sum(exp_values)
    return np.random.choice(len(Q[state_index]), p=probabilities)

# Epsilon-Greedy Stratejisi
def epsilon_greedy(Q, state_index, epsilon):
    if np.random.rand() < epsilon:
        # Keşif: Rastgele bir eylem seç
        action_index = np.random.choice(len(Q[state_index]))
        return action_index, 'explore'
    else:
        # Sömürü: En yüksek Q-değerine sahip eylemi seç
        action_index = np.argmax(Q[state_index])
        return action_index, 'exploit'

# Hibrit Strateji ile Epsilon-Greedy ve Softmax
def hybrid_strategy(Q, state_index, epsilon, temperature, strategy_state):
    if strategy_state['last_strategy'] == 'epsilon_greedy':
        # Epsilon-Greedy stratejisinde 3 kez art arda keşif veya sömürü yapılırsa Softmax'a geç
        if strategy_state['explore_count'] >= 3 or strategy_state['exploit_count'] >= 3:
            action_index = softmax(Q, state_index, temperature)
            strategy = 'softmax'
            strategy_state['last_strategy'] = strategy
            strategy_state['explore_count'] = 0
            strategy_state['exploit_count'] = 0
            return action_index, strategy
        else:
            action_index, strategy = epsilon_greedy(Q, state_index, epsilon)
            # Güncelleme
            if strategy == 'explore':
                strategy_state['explore_count'] += 1
                strategy_state['exploit_count'] = 0
            else:
                strategy_state['exploit_count'] += 1
                strategy_state['explore_count'] = 0
            return action_index, strategy

    elif strategy_state['last_strategy'] == 'softmax':
        # Softmax stratejisinde 3 kez art arda keşif veya sömürü yapılırsa Epsilon-Greedy'ye geç
        if strategy_state['explore_count'] >= 3 or strategy_state['exploit_count'] >= 3:
            action_index, strategy = epsilon_greedy(Q, state_index, epsilon)
            strategy_state['last_strategy'] = 'epsilon_greedy'
            strategy_state['explore_count'] = 0
            strategy_state['exploit_count'] = 0
            return action_index, strategy
        else:
            action_index = softmax(Q, state_index, temperature)
            # Güncelleme (Softmax genellikle yalnızca sömürü yapar)
            strategy_state['explore_count'] = 0
            strategy_state['exploit_count'] += 1
            return action_index, 'softmax'

    else:
        # Başlangıçta Epsilon-Greedy olarak başla
        action_index, strategy = epsilon_greedy(Q, state_index, epsilon)
        strategy_state['last_strategy'] = 'epsilon_greedy'
        strategy_state['explore_count'] = 1 if strategy == 'explore' else 0
        strategy_state['exploit_count'] = 1 if strategy == 'exploit' else 0
        return action_index, strategy

# Q-Tablosu ve Parametreler
num_actions = len(actions)
num_states = 1  # Başlangıçta sadece 1 tane rastgele durum var, durum sayısını artırmanız gerekebilir
Q = np.zeros((num_states, num_actions))
epsilon = 0.1
temperature = 1.0
# Öğrenme oranını tanımla
learning_rate = 0.1

# Strateji durumu
strategy_state = {'last_strategy': None, 'explore_count': 0, 'exploit_count': 0}

# Sonuçları saklamak için liste
results = []

# Simülasyon
for episode in range(5000):  # 10000 episode örneği
    state = generate_random_conditions()
    state_index = 0  # Durum dizini (örneğin: 0)
    
    # Hibrit strateji seçimi
    action_index, strategy = hybrid_strategy(Q, state_index, epsilon, temperature, strategy_state)
    selected_action = actions[action_index]
    
    # Ödül hesaplama
    total_reward = reward_function(state)
    reward_points = reward_points_based_on_action(selected_action, total_reward)
    
     
    
       # Q-değerini güncelleme (öğrenme oranını dahil etme)
    Q[state_index, action_index] = (1 - learning_rate) * Q[state_index, action_index] + learning_rate * reward_points


    # Sonuçları ekleme
    results.append({
        'Episode': episode,
        'State': state,
        'State Index': state_index,
        'Selected Action': selected_action,
        'Strategy Used': strategy,
        'Reward': total_reward,
        'Reward Points': reward_points,
        'Q-value': Q[state_index, action_index]
    })

# DataFrame'e dönüştürme
df_results = pd.DataFrame(results)

# Dizini kontrol et ve oluştur
save_directory = '/content/drive/My Drive'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# CSV dosyasına kaydetme
output_path = os.path.join(save_directory, 'valle.csv')
df_results.to_csv(output_path, index=False)

print("Sonuçlar başarıyla kaydedildi!")

 # Epsilon ve öğrenme oranını ekrana yazdırma
print(f"Son epsilon değeri: {epsilon}")
print(f"Öğrenme oranı: {learning_rate}")
