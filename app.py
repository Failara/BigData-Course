import pandas as pd
import numpy as np
import joblib
import os
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_assets():
    """Завантаження моделі, скейлера та списку ознак."""
    files = ['heart_disease_model.pkl', 'scaler.pkl', 'model_features.pkl']
    
    # Перевірка наявності файлів
    for f in files:
        if not os.path.exists(f):
            print(f"[ПОМИЛКА] Файл '{f}' не знайдено!")
            print("Будь ласка, спочатку запустіть Jupyter Notebook для тренування та збереження моделі.")
            input("Натисніть Enter для виходу...")
            sys.exit()

    print("[INFO] Завантаження моделі та параметрів...")
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_features = joblib.load('model_features.pkl')
    print("[INFO] Завантаження завершено успішно.\n")
    return model, scaler, model_features

def get_user_input_choice(prompt, options):
    """Допоміжна функція для вибору варіанту зі списку."""
    print(f"\n{prompt}")
    for k, v in options.items():
        print(f"  {k}: {v}")
    
    while True:
        choice = input("Ваш вибір (введіть номер): ").strip()
        if choice in options:
            return options[choice]
        print("Невірний вибір. Спробуйте ще раз.")

def get_user_input_numeric(prompt, min_val=0, max_val=1000):
    """Допоміжна функція для введення чисел."""
    while True:
        try:
            val_str = input(f"\n{prompt}: ").strip()
            val = float(val_str)
            if min_val <= val <= max_val:
                return val
            else:
                print(f"Будь ласка, введіть число від {min_val} до {max_val}.")
        except ValueError:
            print("Це не схоже на число. Спробуйте ще раз.")

def main():
    clear_screen()
    print("========================================================")
    print("   СИСТЕМА ДІАГНОСТИКИ СЕРЦЕВО-СУДИННИХ ЗАХВОРЮВАНЬ")
    print("   (на основі Machine Learning)")
    print("========================================================")

    # 1. Завантаження ресурсів
    model, scaler, model_features = load_assets()

    while True:
        print("\n--- Введіть клінічні показники пацієнта ---")
        
        # 2. Збір даних від користувача
        age = get_user_input_numeric("Вік (років)", 10, 100)
        
        sex_map = {'1': 'Male', '2': 'Female'}
        sex_choice = get_user_input_choice("Стать:", sex_map)
        sex_val = 1 if sex_choice == 'Male' else 0

        trestbps = get_user_input_numeric("Артеріальний тиск у спокої (мм рт.ст.)", 80, 250)
        chol = get_user_input_numeric("Холестерин сироватки (мг/дл)", 100, 600)
        
        fbs_map = {'1': True, '2': False}
        fbs_choice = get_user_input_choice("Рівень цукру в крові > 120 мг/дл?", fbs_map)
        fbs_val = 1 if fbs_choice else 0

        thalch = get_user_input_numeric("Максимальна досягнута частота серцевих скорочень", 60, 220)
        
        exang_map = {'1': True, '2': False}
        exang_choice = get_user_input_choice("Стенокардія викликана фіз. навантаженням?", exang_map)
        exang_val = 1 if exang_choice else 0

        oldpeak = get_user_input_numeric("Депресія ST (oldpeak), наприклад 2.5", 0.0, 10.0)
        
        ca = get_user_input_numeric("Кількість великих судин (0-3), забарвлених флюороскопією", 0, 4)

        # Категоріальні ознаки (для One-Hot Encoding)
        # Важливо: значення повинні точно співпадати з тими, що були в dataset
        cp_options = {
            '1': 'typical angina', 
            '2': 'atypical angina', 
            '3': 'non-anginal', 
            '4': 'asymptomatic'
        }
        cp_val = get_user_input_choice("Тип болю в грудях:", cp_options)

        restecg_options = {
            '1': 'normal', 
            '2': 'st-t abnormality', 
            '3': 'lv hypertrophy'
        }
        restecg_val = get_user_input_choice("ЕКГ спокою:", restecg_options)

        slope_options = {
            '1': 'upsloping', 
            '2': 'flat', 
            '3': 'downsloping'
        }
        slope_val = get_user_input_choice("Нахил пікового сегмента ST (Slope):", slope_options)

        thal_options = {
            '1': 'normal', 
            '2': 'fixed defect', 
            '3': 'reversable defect'
        }
        thal_val = get_user_input_choice("Таласемія (Thal):", thal_options)

        # 3. Формування DataFrame для моделі
        # Створюємо порожній DataFrame з усіма колонками, які очікує модель
        input_data = pd.DataFrame(columns=model_features)
        input_data.loc[0] = 0  # Ініціалізуємо нулями

        # Заповнюємо числові та бінарні дані
        input_data['age'] = age
        input_data['sex'] = sex_val
        input_data['trestbps'] = trestbps
        input_data['chol'] = chol
        input_data['fbs'] = fbs_val
        input_data['thalch'] = thalch
        input_data['exang'] = exang_val
        input_data['oldpeak'] = oldpeak
        input_data['ca'] = ca

        # Обробка One-Hot Encoding
        # Ми перевіряємо, чи існує колонка виду "ознака_значення" у моделі.
        # Якщо так - ставимо 1. Якщо ні (вона була drop_first) - залишаємо всі 0 (базовий випадок).
        
        cat_inputs = {
            'cp': cp_val,
            'restecg': restecg_val,
            'slope': slope_val,
            'thal': thal_val
        }

        for feature, value in cat_inputs.items():
            col_name = f"{feature}_{value}"
            if col_name in input_data.columns:
                input_data[col_name] = 1

        # 4. Масштабування (Scaling)
        # Використовуємо той самий scaler, що і при навчанні
        input_scaled = scaler.transform(input_data)

        # 5. Прогноз
        prediction = model.predict(input_scaled)[0]
        
        # Спробуємо отримати ймовірність, якщо модель це підтримує
        probability = "N/A"
        if hasattr(model, "predict_proba"):
            prob_val = model.predict_proba(input_scaled)[0][1]
            probability = f"{prob_val:.2%}"

        # 6. Виведення результату
        print("\n" + "="*40)
        print("          РЕЗУЛЬТАТ ДІАГНОСТИКИ")
        print("="*40)
        
        if prediction == 1:
            print(f"  [!] Ризик ССЗ: ВИСОКИЙ (Є хвороба)")
            print(f"  Впевненість моделі: {probability}")
            print("  Рекомендація: Звернутися до кардіолога.")
        else:
            print(f"  [OK] Ризик ССЗ: НИЗЬКИЙ (Здоровий)")
            print(f"  Впевненість моделі: {probability}")
        print("="*40 + "\n")

        # Питання про повтор
        retry = input("Бажаєте перевірити іншого пацієнта? (y/n): ").lower()
        if retry != 'y':
            print("Дякуємо за використання програми. До побачення!")
            break
        
        clear_screen()

if __name__ == "__main__":
    main()