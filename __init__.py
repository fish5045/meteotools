from time import time

def timer(func):
    def func_wrapper(*args, **kwargs):
        t0 = time()
        result = func(*args, **kwargs)
        t1 = time()
        print(f'{func.__name__} cost time: {(t1-t0):.5f} (s).')
        return result
    return func_wrapper

#version: 0.1.3
# 新增 calc.FD2 

#version 0.1.2 2021/10/23
# 新增 calc.calc_Td
# 新增 calc.calc_theta_v
# 修正 calc.xy2lonlat 與 calc.lonlat2xy
# 修改 calc.calc_vapor 可計算水氣壓與飽和水氣壓
# 修改 calc.calc_qv 可用Td計算qv

#version 0.1.1
# calc 將預設之熱力參數由np.nan改為[]
# 新增 calc.calc_T
# 新增 calc.calc_Tc
# 新增 calc.calc_theta_es
# 修正 calc.calc_theta_e

#version 0.1
# - 新增 calc.ZtoP
# - 新增 calc.calc_dTdz