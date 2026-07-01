import numpy as np


# Угол между скоростью и магнитным полем, радианы
def get_alpha(B, V):
    cos_BV = np.dot(B, V) / (np.linalg.norm(B) * np.linalg.norm(V))
    alpha = np.arccos(cos_BV)
    return alpha


# Норма перпендикулярной к скорости проекции вектора магнитного поля
def get_norm_B_perp(B, V):
    norm_B = np.linalg.norm(B)
    alpha = get_alpha(B, V)
    return norm_B * np.sin(alpha)


class SynchCounter:
    def __init__(self):
        self.T = 0  # Кинетическая энергия
        self.norm_B_perp = 0  # Норма перпендикулярной к скорости проекции вектора магнитного поля
        self.delta_t = 0  # Промежуток времени
        self.records = 0  # Количество записей

    def add_iteration(self, T, B, V, dt):
        self.T += T
        self.norm_B_perp += get_norm_B_perp(B, V)
        self.delta_t += dt
        self.records += 1

    def get_averages(self):
        T_avg = self.T / self.records
        B_perp_avg = self.norm_B_perp / self.records
        return T_avg, B_perp_avg
    
    def print(self):
        print(f"T: {self.T}, norm_B_perp: {self.norm_B_perp}, delta_t: {self.delta_t}, records: {self.records}")

    def reset(self):
        self.__init__()