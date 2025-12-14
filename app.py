import joblib
import numpy as np
import os

def load_assets():
    """Завантажує навчену модель та скейлер."""
    if not os.path.exists('heart_disease_model.pkl') or not os.path.exists('scaler.pkl'):
        print("Помилка: Файли моделі не знайдено. Спочатку запустіть навчання.")
        return None, None
    
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

def get_user_input():
    """Отримує дані від користувача через консоль."""
    print("\n--- Введіть клінічні показники пацієнта ---")
    
    try:
        age = float(input("Вік (років): "))
        sex = float(input("Стать (1 = чол, 0 = жін): "))
        cp = float(input("Тип болю в грудях (0-3): "))
        trestbps = float(input("Артеріальний тиск у спокої (мм рт.ст.): "))
        chol = float(input("Холестерин (мг/дл): "))
        fbs = float(input("Цукор в крові > 120 мг/дл (1 = так, 0 = ні): "))
        restecg = float(input("ЕКГ у спокої (0-2): "))
        thalach = float(input("Максимальний пульс: "))
        exang = float(input("Стенокардія при навантаженні (1 = так, 0 = ні): "))
        oldpeak = float(input("Депресія ST (наприклад, 2.5): "))
        slope = float(input("Нахил сегмента ST (0-2): "))
        ca = float(input("Кількість великих судин (0-3): "))
        thal = float(input("Таласемія (1 = норм, 2 = фіксований дефект, 3 = оборотний): "))
        
        # Формуємо масив даних (порядок має збігатися з тренуванням!)
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        return features
    except ValueError:
        print("Помилка: Будь ласка, вводьте тільки цифрові значення.")
        return None

def main():
    print("=== Система діагностики ризику серцево-судинних захворювань ===")
    
    model, scaler = load_assets()
    if not model:
        return

    while True:
        features = get_user_input()
        
        if features is not None:
            # 1. Масштабування вхідних даних
            features_scaled = scaler.transform(features)
            
            # 2. Прогноз
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            
            # 3. Вивід результату
            print("\n" + "="*30)
            if prediction == 1:
                print(f"!!! РИЗИК ССЗ: ВИСОКИЙ !!!")
                print(f"Ймовірність: {probability:.2%}")
                print("Рекомендація: Негайно звернутися до кардіолога.")
            else:
                print(f"Ризик ССЗ: НИЗЬКИЙ")
                print(f"Ймовірність хвороби: {probability:.2%}")
                print("Рекомендація: Продовжуйте вести здоровий спосіб життя.")
            print("="*30 + "\n")
        
        cont = input("Бажаєте перевірити ще одного пацієнта? (y/n): ")
        if cont.lower() != 'y':
            break

if __name__ == "__main__":
    main()