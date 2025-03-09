from netCDF4 import Dataset
import numpy as np
import wrf
from .grid_general import gridsystem
from ..fileprocess import get_time_sec, get_wrfout_time
from ..tools import timer


class wrfout_grid(gridsystem):
    '''
    wrfout 的網格系統，3維網格(z, y, x)，x, y為等間距的直角水平座標，z為eta座標
    '''

    def init_grid(self):
        self.offset = 0

        # 設定由josn檔案來，其數字不分整數或浮點數，由這兩個function強制區分
        self.meta_convert_to_float()
        self.meta_convert_to_int('start_yr', 'start_mo', 'start_dy',
                                 'start_hr', 'start_mi', 'start_sc',
                                 'end_yr', 'end_mo', 'end_dy',
                                 'end_hr', 'end_mi', 'end_sc',
                                 'domain')

        # 由設定檔得到的時間資訊，生成對應的wrfout檔名
        self.set_wrfout_fname()

    def set_wrfout_fname(self):
        self.wrfout_time = get_wrfout_time(self.start_yr, self.start_mo,
                                           self.start_dy, self.start_hr,
                                           self.start_mi, self.start_sc,
                                           self.offset)
        self.time = get_time_sec(self.start_yr, self.start_mo,
                                 self.start_dy, self.start_hr,
                                 self.start_mi, self.start_sc,
                                 self.offset)
        self.wrfout_fname = \
            f"{self.wrfout_dir}/{self.wrf_prefix}d{self.domain:02d}_{self.wrfout_time}"

    def set_offset(self, second):
        '''
        設定相對於設定wrfout時間，要取偏移多少時間的wrfout
        (可用於取local tendency需要取的前後時間點的wrfout)
        '''
        self.offset = second
        self.set_wrfout_fname()

    def correct_map_scale(self):
        self.ua *= self.MAPFAC_MX
        self.va *= self.MAPFAC_MY
        try:
            self.u10 *= self.MAPFAC_MX
            self.v10 *= self.MAPFAC_MY
        except:
            pass

    @timer
    def read_data(self, *varlist):
        '''
        讀取wrfout資料，可給定wrf-python的getvar支援的變數（將內插到A grid)
        或是wrfout自身的變數(原始C grid位置)
        讀取wrfout網格資訊
        '''
        with Dataset(self.wrfout_fname) as filenc:
            self.dx = filenc.DX
            self.dy = filenc.DY
            self.Nx = filenc.dimensions['west_east'].size
            self.Ny = filenc.dimensions['south_north'].size
            self.center_xloc = ((self.Nx+1)/2 - 1) * self.dx
            self.center_yloc = ((self.Ny+1)/2 - 1) * self.dy

            for var in varlist:
                try:
                    exec(
                        f'self.{var} = np.squeeze(np.array(wrf.getvar(filenc,"{var}")))')
                    if var == 'P_HYD':
                        self.pressure = self.P_HYD
                        self.sfc_pressure = self.pressure[0]
                        #exec(f'self.{var} = wrf.smooth2d(self.{var},2)')

                except ValueError:
                    exec(
                        f'self.{var} = np.squeeze(np.array(filenc.variables["{var}"]))')

                except:
                    raise ValueError(
                        f"Variable name '{var}' is not available in this nc file.")
