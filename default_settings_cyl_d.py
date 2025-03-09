import json
from pathlib import Path
import numpy as np

d = dict()
d['wrf'] = dict()
d['cylindrical_grid'] = dict()
d['cartesian_grid'] = dict()
d['system'] = dict()

# ------------------------------------- #
#                settings               #
# ------------------------------------- #

dwrf = d['wrf']  # wrfout 設定
dwrf['wrfout_dir'] = r'D:\Research_data\WRF_data\LEKIMA4_new_WDM6_5min'  # wrfout 位置
dwrf['start_yr'] = 2019  # wrfout 診斷起始年
dwrf['start_mo'] = 8  # 月
dwrf['start_dy'] = 8  # 日
dwrf['start_hr'] = 13  # 十
dwrf['start_mi'] = 0  # 分
dwrf['start_sc'] = 0  # 秒
dwrf['end_yr'] = 2019  # 終止年
dwrf['end_mo'] = 8  # 月
dwrf['end_dy'] = 8  # 日
dwrf['end_hr'] = 13  # 時
dwrf['end_mi'] = 0  # 分
dwrf['end_sc'] = 0  # 秒
dwrf['domain'] = 4  # domain編號
dwrf['dt'] = 3600  # 2wrfout的時間間格dt

#直角坐標系網格設定 (用於疊代擾動氣壓)
dcyl = d['cylindrical_grid']  # 圓柱座標設定
dcyl['Nr'] = 300  # 徑向格數
dcyl['Ntheta'] = 90  # 切向格數
dcyl['Nz'] = 15  # 垂直格數
dcyl['dr'] = 1000  # 徑向網格大小 (m)
dcyl['dtheta'] = 2*np.pi/dcyl['Ntheta']  # 切向網格大小 (rad)
dcyl['dz'] = 1000  # 垂直網格大小 (m)
dcyl['z_start'] = 100  # 垂直起始位置 (m)
dcyl['r_start'] = 0  # 徑向起始位置 (m)
dcyl['theta_start'] = 0  # 切向起始位置 (rad)
dcyl['vertical_coord_type'] = 'z'  # 垂直座標
dcyl['dt'] = 300  # tendency 使用的dt

# 直角座標系網格設定 (用於計算perturbation pressure)
dcar = d['cartesian_grid']  # 直角坐標
dcar['Nx'] = 350  # x方向網格數
dcar['Ny'] = 350  # y方向網格數
dcar['Nz'] = 192  # z方向網格數
dcar['dx'] = 2000  # x方向網格大小
dcar['dy'] = 2000  # y方向網格大小
dcar['dz'] = 120  # z方向網格大小
dcar['z_start'] = 60  # 垂直起始位置 (m)
dcar['vertical_coord_type'] = 'z'  # 垂直座標
dcar['x_boundry_west'] = 'Neumann'  # x方向邊界條件(西側)
dcar['x_boundry_east'] = 'Neumann'  # x方向邊界條件(東側)
dcar['y_boundry_south'] = 'Neumann'  # y方向邊界條件(南側)
dcar['y_boundry_north'] = 'Neumann'  # y方向邊界條件(北側)
dcar['z_boundry_bottom'] = 'Neumann'  # z方向邊界條件(底部)
dcar['z_boundry_top'] = 'Zero'  # z方向邊界條件(頂部)
dcar['SOR_parameter'] = 1.947  # SOR 收斂係數
dcar['max_iteration_times'] = 2501  # SOR 收斂係數
dcar['iteration_tolerance'] = 5e-2  # SOR 收斂係數


# 系統相關設定
dsys = d['system']
dsys['calc_cpus'] = 8
dsys['interp_cpus'] = 8


Path('./settings').mkdir(parents=True, exist_ok=True)
with open('./settings/settings_cyl_d.json', 'w') as f:
    f.write(json.dumps(d, indent=4))
