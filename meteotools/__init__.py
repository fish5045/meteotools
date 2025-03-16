import inspect
import platform
import os

frame = inspect.currentframe()
file_path = frame.f_globals['__file__']
path_of_meteotools = os.path.dirname(os.path.abspath(file_path))
platform_name = platform.system()

# version: 0.2.1 2023/3/14
# 新增calc.grids (尚未測試)

# version: 0.2 2023/3/10
# 新增calc.averages
# 新增calc.check_arrays
# 新增calc.coordinates
# 修改 calc.FD2 -> calc.differential
# 新增calc.math
# 新增calc.thermaldynamic
# 新增calc.transformation
# 新增interpolation.check_arrays
# 新增interpolation.interp_1d
# 新增interpolation.interp_2d
# 新增interpolation.interp_3d

# version: 0.1.3
# 新增 calc.FD2

# version 0.1.2 2021/10/23
# 新增 calc.calc_Td
# 新增 calc.calc_theta_v
# 修正 calc.xy2lonlat 與 calc.lonlat2xy
# 修改 calc.calc_vapor 可計算水氣壓與飽和水氣壓
# 修改 calc.calc_qv 可用Td計算qv

# version 0.1.1
# calc 將預設之熱力參數由np.nan改為[]
# 新增 calc.calc_T
# 新增 calc.calc_Tc
# 新增 calc.calc_theta_es
# 修正 calc.calc_theta_e

# version 0.1
# - 新增 calc.ZtoP
# - 新增 calc.calc_dTdz
