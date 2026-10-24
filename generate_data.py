import numpy as np
from scipy.optimize import minimize
import time

# ==========================================
# 1. АНАЛИТИЧЕСКИЕ ФУНКЦИИ СРЕДЫ
# ==========================================

def E_func(x, eps1, C1, sigma1):
    #Пища: сигмоидальный спад с глубиной
    # x <= 0 (глубина в метрах со знаком минус)
    return sigma1 * (np.tanh(eps1 * x + C1) + 1.0)

def Sx_func(x, eps2, C2, sigma2):
    #Хищник по глубине: концентрация в верхних освещенных слоях
    return sigma2 * (np.tanh(eps2 * x + C2) + 1.0)

def St_func(t):
    #Суточный ход хищника: максимум днем (t=0.5), минимум ночью (t=0, t=1)
    # -cos(2*pi*t) дает минимум в полночь и максимум в полдень
    return -np.cos(2.0 * np.pi * t) + 1.0

def G_func(x, eps3, C0, xi):
    #Стресс (температура + H2S): минимум на оптимуме ХПС (-C0), рост вверх и вниз
    arg = np.clip(eps3 * (x + C0), -15.0, 15.0)  # защита от переполнения exp
    return xi * np.cosh(arg)

# ==========================================
# 2. РАСЧЕТ ИНТЕГРАЛОВ ПРИСПОСОБЛЕННОСТИ
# ==========================================

# Сетка времени для численного интегрирования за сутки
T_STEPS = 60
t_grid = np.linspace(0.0, 1.0, T_STEPS)
st_grid = St_func(t_grid)

def compute_fitness(v, env_params, weights):
    # Вычисляет J(A, B) для стратегии v = (A, B)
    A, B = v[0], v[1]
    
    # Жесткий штраф за выход за пределы водного столба:
    # планктон не может вылетать в воздух (> 0) и падать глубже шельфа (< -140)
    if (A + B > 0.0) or (A - B < -140.0) or (B < 0.0):
        return -1e6

    # Траектория движения особи в течение суток
    # Ночью (t=0, 1) планктон у поверхности: x = A + B
    # Днем (t=0.5) планктон на глубине: x = A - B
    x_traj = A + B * np.cos(2.0 * np.pi * t_grid)
    
    # 1. Пища M1 (интеграл по суткам)
    e_vals = E_func(x_traj, env_params['eps1'], env_params['C1'], env_params['sigma1'])
    M1 = np.trapezoid(e_vals, t_grid)
    
    # 2. Риск выедания M2 (штраф)
    sx_vals = Sx_func(x_traj, env_params['eps2'], env_params['C2'], env_params['sigma2'])
    M2 = -np.trapezoid(st_grid * sx_vals, t_grid)
    
    # 3. Энергетические затраты M3 = -2 * pi^2 * B^2
    M3 = -2.0 * (np.pi ** 2) * (B ** 2) * 1e-4  # масштабирование затрат движения
    
    # 4. Урон от условий G M4 (штраф)
    g_vals = G_func(x_traj, env_params['eps3'], env_params['C0'], env_params['xi'])
    M4 = -np.trapezoid(g_vals, t_grid)
    
    alpha, beta, gamma, delta = weights
    J = alpha * M1 + gamma * M2 + beta * M3 + delta * M4
    return J

# ==========================================
# 3. ГЕНЕРАТОР ВЫБОРКИ
# ==========================================

def generate_dataset(num_samples=1000, noise_level=0.05, migration_threshold=20.0):
    # Генерирует матрицу X (num_samples, 84) и метки Y (num_samples)
    depth_sensors = np.linspace(0.0, -140.0, 20)  # 20 горизонтов глубины
    time_sensors  = np.linspace(0.0, 1.0, 20)      # 20 отсчетов времени
    
    X_list = []
    Y_list = []
    B_opt_list = []
    
    print(f"Генерация {num_samples} обучающих примеров...")
    start_time = time.time()
    
    for i in range(num_samples):
        # 1. Разыгрываем экологические приоритеты популяции
        alpha = np.random.uniform(0.8, 1.2)
        beta  = np.random.uniform(0.03, 0.07)
        gamma = np.random.uniform(0.5, 1.2)
        delta = np.random.uniform(0.8, 1.4)
        weights = (alpha, beta, gamma, delta)
        
        # 2. Разыгрываем параметры гидрологической структуры моря
        # Слой оптимума ХПС (холодного промежуточного слоя): 60-80 м
        C0 = np.random.uniform(60.0, 80.0)

        # можно задать значения как массив фиксированных чисел. Что лучше??
        
        env_params = {
            'eps1': np.random.uniform(0.04, 0.07),
            'C1':   np.random.uniform(1.5, 2.5),
            'sigma1': np.random.uniform(0.8, 1.2),
            
            'eps2': np.random.uniform(0.04, 0.08),
            'C2':   np.random.uniform(1.2, 2.2),
            'sigma2': np.random.uniform(0.6, 1.4),
            
            'eps3': np.random.uniform(0.03, 0.06),
            'C0':   C0,
            'xi':   np.random.uniform(0.1, 0.3)
        }
        
        # 3. Снятие 80 точек датчиков среды
        E_discrete  = E_func(depth_sensors, env_params['eps1'], env_params['C1'], env_params['sigma1'])
        Sx_discrete = Sx_func(depth_sensors, env_params['eps2'], env_params['C2'], env_params['sigma2'])
        St_discrete = St_func(time_sensors)
        G_discrete  = G_func(depth_sensors, env_params['eps3'], env_params['C0'], env_params['xi'])
        
        # Наложение белого шума датчиков (5-10%)

        # как будто не обязателен

        if noise_level > 0.0:
            E_discrete  *= (1.0 + np.random.uniform(-noise_level, noise_level, size=20))
            Sx_discrete *= (1.0 + np.random.uniform(-noise_level, noise_level, size=20))
            St_discrete *= (1.0 + np.random.uniform(-noise_level, noise_level, size=20))
            G_discrete  *= (1.0 + np.random.uniform(-noise_level, noise_level, size=20))
            
        # Формирование вектора X из 84 признаков
        x_vector = np.concatenate([
            E_discrete, 
            Sx_discrete, 
            St_discrete, 
            G_discrete, 
            [alpha, beta, gamma, delta]
        ])
        
        # 4. МОДЕЛЬ ОТБОРА: поиск оптимальной стратегии (A*, B*)
        def loss_to_minimize(v, env_params, weights):
            return -compute_fitness(v, env_params, weights)
        
        # Начальная точка поиска: глубина оптимума ХПС, амплитуда 15 м
        init_guess = [-C0, 15.0]
        
        opt_res = minimize(
            loss_to_minimize, 
            x0=init_guess, 
            args=(env_params, weights),
            method='Nelder-Mead',
            options={'maxiter': 150, 'xatol': 0.5, 'fatol': 0.01}
        )
        
        A_opt, B_opt = opt_res.x
        B_opt = max(0.0, float(B_opt))
        
        # 5. Разметка целевого класса Y
        label = 1 if B_opt >= migration_threshold else 0
        
        X_list.append(x_vector)
        Y_list.append(label)
        B_opt_list.append(B_opt)
        
        if (i + 1) % 100 == 0 or (i + 1) == num_samples:
            print(f"Обработано {i + 1}/{num_samples} примеров (время: {time.time() - start_time:.1f} сек)")
            
    X_arr = np.array(X_list, dtype=np.float32)
    Y_arr = np.array(Y_list, dtype=np.int64)
    
    return X_arr, Y_arr, np.array(B_opt_list)

# ==========================================
# 4. ПРИМЕР ЗАПУСКА И СОХРАНЕНИЯ
# ==========================================
if __name__ == "__main__":
    # Для теста генерируем 200 примеров (для итоговой выборки укажите 10000)
    X, Y, B_values = generate_dataset(num_samples=200, noise_level=0.05, migration_threshold=20.0)
    
    print("\nСвойства сгенерированного датасета:")
    print(f"Размерность матрицы X: {X.shape} (должно быть N x 84)")
    print(f"Размерность вектора Y:  {Y.shape}")
    print(f"Класс 1 (миграция есть): {np.sum(Y == 1)} шт.")
    print(f"Класс 0 (миграции нет):  {np.sum(Y == 0)} шт.")
    
    # Сохранение в стандартный файл NumPy для последующего обучения перцептрона
    np.savez_compressed("dvm_dataset.npz", X=X, Y=Y, B=B_values)
    print("Датасет успешно сохранен в файл 'dvm_dataset.npz'")