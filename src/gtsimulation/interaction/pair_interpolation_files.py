import numpy as np
from scipy.special import kv
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d, RegularGridInterpolator

prob_grid = np.linspace(0, 1, 1000)
chi_grid = np.logspace(-1, 20, 1000)
epsilon_table = np.zeros((len(chi_grid), len(prob_grid)))
cdf_table = np.zeros((len(chi_grid), len(prob_grid)))

# Сетка для интегрирования (от 0 до 0.5 по симметрии)
e_integration_grid = np.linspace(1e-5, 0.5, 500)

for i, chi in enumerate(chi_grid):
    term_bessel = 1.0 / (3.0 * chi * e_integration_grid * (1.0 - e_integration_grid))
    factor = (2.0 + e_integration_grid * (1.0 - e_integration_grid)) / (e_integration_grid * (1.0 - e_integration_grid))
    pdf_values = factor * kv(2/3, term_bessel)
    
    cdf_values = cumulative_trapezoid(pdf_values, e_integration_grid, initial=0)
    total_sum = cdf_values[-1]

    if total_sum > 1e-100: # используем очень малое число вместо 0
        cdf_values /= total_sum
        
        # берем только те индексы, где CDF растет
        # это убирает плоские участки которые не дают работать interp1d
        _, idx = np.unique(cdf_values, return_index=True)
        
        # для интерполяции нам нужно сохранить исходный порядок сетки по e_integration_grid
        idx = np.sort(idx)

        # проверяем, что уникальных точек хотя бы 2, иначе интерполяция невозможна
        if len(idx) > 1:
            inv_func = interp1d(cdf_values[idx], 
                                e_integration_grid[idx], 
                                kind='linear', fill_value="extrapolate")
            epsilon_table[i, :] = inv_func(prob_grid)
        else:
            # затычка/костыль
            epsilon_table[i, :] = 0.5

if not np.isfinite(epsilon_table).all():
    print("В таблице все еще есть NaN!")

# for row in cdf_table[:100]:
#     print("".join(f"{item==0}" for item in row))

CDF_Interpolator = RegularGridInterpolator(
                points=(np.log10(chi_grid), prob_grid),
                values=epsilon_table,
                method='cubic',
                bounds_error=False,
                fill_value=0.5         # значение по умолчанию при выходе за границы
            )
np.savez_compressed(
    "epsilon_interpolation_data.npz", 
    epsilon_table=epsilon_table,
    chi_grid=chi_grid,
    prob_grid=prob_grid
)

print("Создана и сохранена таблица интерполяции эпсилон")