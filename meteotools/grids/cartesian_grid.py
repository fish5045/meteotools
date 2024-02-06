from scipy.interpolate import splev, splrep
import numpy as np
import wrf
import multiprocessing as mp
from ..interpolation import interp2D_fast_layers, interp2D_fast
from ..calc import FD2, calc_Zaverage
from ..exceptions import DimensionError
from .grid_general import gridsystem
from ..tools import timer
from .thermaldynamic_calculation import calc_thermaldynamic
from scipy.interpolate import interpn

class cartesian_grid(gridsystem, calc_thermaldynamic):
    '''
    圓柱座標網格，3維網格(z, y, x)，z可為壓力或高度座標，x, y為水平直角等間距座標。
    '''

    def init_grid(self):
        self.meta_convert_to_float('dx', 'dy', 'dz', 'z_start',
                                   'SOR_parameter', 'iteration_tolerance')
        self.meta_convert_to_int('Nx', 'Ny', 'Nz', 'max_iteration_times')

        # 設定x, y, z座標軸，及中心位置。
        self.create_coord()
        self.varlist3D = []
        self.varlist2D = []

    def make_center_loc(self):
        self.center_xloc = (self.x[0] + self.x[-1])/2
        self.center_yloc = (self.y[0] + self.y[-1])/2

    def create_coord(self):
        self.x = np.arange(0., self.Nx*self.dx, self.dx)
        self.y = np.arange(0., self.Ny*self.dy, self.dy)
        self.z = np.arange(0, self.Nz*self.dz, self.dz) + self.z_start
        self.make_center_loc()
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.dis_from_center = (
            self.xx-self.center_xloc)**2 + (self.yy-self.center_yloc)**2

        self.set_coord_meta('z', 'z', self.z, unit='m',
                            description='Z coordinate')
        self.set_coord_meta('y', 'y', self.y, unit='m',
                            description='Y coordinate')
        self.set_coord_meta('x', 'x', self.x, unit='m',
                            description='X coordinate')

    def set_horizontal_location(self, offset_x, offset_y):
        '''
        設定內插時的目標位置
        來源資料可能是直角坐標或是wrf座標，設定此座標系的中心要在來源座標系的哪裡
        (offset_x, offset_y)，產生出此直角坐標系在來源座標系內的x, y位置，供內插使用
        '''
        if 'center_xloc' not in dir(self) or 'center_yloc' not in dir(self):
            self.make_center_loc()
        self.hinterp_target_xloc = self.x + offset_x - self.center_xloc
        self.hinterp_target_yloc = self.y + offset_y - self.center_yloc
        self.hinterp_target_xxloc, self.hinterp_target_yyloc = \
            np.meshgrid(self.hinterp_target_xloc, self.hinterp_target_yloc)

    def spline(self, var, vert, obj_vert, order=2):
        '''
        針對網格垂直方向使用高階spline內插（預設order=2)
        '''
        Nx = var.shape[2]
        Ny = var.shape[1]
        out = np.zeros([len(obj_vert), Ny, Nx])

        for j in range(Ny):
            for i in range(Nx):
                cs = splrep(vert[:, j, i], var[:, j, i], k=order)
                out[:, j, i] = splev(obj_vert, cs)

        return out

    @timer
    def set_data(self, src_dx, src_dy, src_vertical,
                 vertical_interp_order = 1, **vardict):
        '''
        將來源資料(vardict)內插到此網格座標系後，結果存於此網格
        需給定來源網格的src_dx, src_dy(需為固定單一值)，垂直位置(src_vertical)
        首次內插的變數必須包含hgt地形高度
        '''

        varlist3D_curr = []

        # 先處理2D陣列，3D陣列之後平行化處理
        for varname, var in vardict.items():
            if len(var.shape) == 3:
                self.varlist3D.append(varname)
                varlist3D_curr.append(varname)
            elif len(var.shape) == 2:
                exec(
                    f'self.{varname} = interp2D_fast(src_dx, src_dy, self.hinterp_target_xxloc, self.hinterp_target_yyloc, var)')
                self.varlist2D.append(varname)

        # 3D平行化垂直內插
        group = [[vardict[varname], src_vertical, self.z, vertical_interp_order]
                 for varname in varlist3D_curr]

        if vertical_interp_order == 1:
            tmp = []
            for var, vertical, specific_z, vertical_interp_order in group:
                tmp.append(np.array(wrf.interplevel(
                    var, vertical, specific_z)))


        elif vertical_interp_order == 2 or vertical_interp_order == 3:
            pool = mp.Pool(self.interp_cpus)
            tmp = pool.starmap(self.spline, group)
            pool.close()
            pool.join()


        # 3D水平內插
        for idx, varname in enumerate(varlist3D_curr):
            tmp2 = interp2D_fast_layers(
                src_dx, src_dy, self.hinterp_target_xxloc, self.hinterp_target_yyloc, tmp[idx])
            exec(f'self.{varname} = tmp2')



        if 'hgt' in self.varlist2D:
            self.set_terrain()  # 設定地形高度
            self.terrain_mask_vars()  # 將地形內部的格點設定為nan

    def set_terrain(self):
        self.z_3d = np.stack([np.stack([self.z]*self.Ny, axis=-1)]*self.Nx, axis=-1)
        self.hgt_3d = np.stack([self.hgt]*self.Nz)
        self.terrain_mask = np.where(self.z_3d <= self.hgt_3d, True, False)


    def terrain_mask_vars(self):
        for varname in self.varlist3D:
            exec(
                f'self.{varname} = np.where(self.terrain_mask,np.nan,self.{varname})')

    # ----------------------------------------------------------------- #
    # computation  (cartesian coordinates only)                         #
    # ----------------------------------------------------------------- #

    def smooth1d(self, varname, axis=0, center_weight=2, passes=1):
        '''
        將指定變數var進行1維的1-center_weight-1平滑
        輸出平滑後變數smooth1d_var

        Parameters
        ----------
        varname : 要進行平滑的變數名稱
        axis : 要進行平滑的維度，預設為第0維
        center_weight : 中心權重，愈大則平滑效果愈弱，預設為2
        passes : 要進行平滑的次數，預設為1
        '''
        var = np.moveaxis(eval(f'self.{varname}'), axis, 0)
        shape = list(var.shape)
        exec(f'self.smooth1d_{varname} = np.zeros(var.shape)')
        shape[0] += 2
        tmp = np.zeros(shape)
        tmp[1:-1] = var
        tmp[0] = var[0]
        tmp[-1] = var[-1]
        for i in range(passes):
            tmp[1:-1] = (tmp[:-2] + tmp[2:] + tmp[1:-1]
                         * center_weight)/(2+center_weight)
        exec(f'self.smooth1d_{varname} = np.moveaxis(tmp[1:-1], 0, axis)')

    def smooth2d(self, varname, axis=(1, 2), center_weight=2, passes=1):
        '''
        將指定變數var進行1維的1-center_weight-1平滑
        輸出平滑後變數smooth1d_var

        Parameters
        ----------
        varname : 要進行平滑的變數名稱
        axis : 要進行平滑的維度，預設為第1、2維
        center_weight : 中心權重，愈大則平滑效果愈弱，預設為2
        passes : 要進行平滑的次數，預設為1
        '''
        var = np.moveaxis(eval(f'self.{varname}'), axis[1], 0)
        var = np.moveaxis(var, axis[0]+1, 0)
        shape = list(var.shape)
        exec(f'self.smooth1d_{varname} = np.zeros(var.shape)')
        shape[0] += 2
        shape[1] += 2
        tmp = np.zeros(shape)
        tmp[1:-1, 1:-1] = var
        tmp[0, 1:-1] = var[0, :]
        tmp[-1, 1:-1] = var[-1, :]
        tmp[1:-1, 0] = var[:, 0]
        tmp[1:-1, -1] = var[:, -1]
        for i in range(passes):
            tmp[1:-1, 1:-1] = (tmp[:-2, 1:-1] + tmp[2:, 1:-1] + tmp[1:-1, :-2]
                               + tmp[1:-1, 2:] + tmp[1:-1, 1:-1]
                               * center_weight)/(4+center_weight)
        exec(f'out = np.moveaxis(tmp[1:-1,1:-1], 0, {axis[0]+1})')
        exec(f'self.smooth2d_{varname} = np.moveaxis(out, 0, {axis[1]})')

    def smooth3d(self, varname, center_weight=2, passes=1):
        '''
        將指定變數var進行1維的1-center_weight-1平滑
        輸出平滑後變數smooth1d_var

        Parameters
        ----------
        varname : 要進行平滑的變數名稱
        center_weight : 中心權重，愈大則平滑效果愈弱，預設為2
        passes : 要進行平滑的次數，預設為1
        '''
        var = eval(f'self.{varname}')
        shape = list(var.shape)
        exec(f'self.smooth1d_{varname} = np.zeros(var.shape)')
        shape[0] += 2
        shape[1] += 2
        shape[2] += 2
        tmp = np.zeros(shape)
        tmp[1:-1, 1:-1, 1:-1] = var
        tmp[0, 1:-1, 1:-1] = var[0, :, :]
        tmp[-1, 1:-1, 1:-1] = var[-1, :, :]
        tmp[1:-1, 0, 1:-1] = var[:, 0, :]
        tmp[1:-1, -1, 1:-1] = var[:, -1, :]
        tmp[1:-1, 1:-1, 0] = var[:, :, 0]
        tmp[1:-1, 1:-1, -1] = var[:, :, -1]
        for i in range(passes):
            tmp[1: -1, 1: -1, 1:-1] = (tmp[: -2, 1: -1, 1: -1] +
                                       tmp[2:, 1: -1, 1: -1] + tmp[1: -1, : -2, 1: -1] +
                                       tmp[1: -1, 2:, 1: -1] + tmp[1: -1, 1: -1, 2:] +
                                       tmp[1: -1, 1: -1, : -2] + tmp[1: -1, 1: -1, 1: -1] *
                                       center_weight) / (6 + center_weight)
        exec(f'self.smooth3d_{varname} = tmp[1:-1,1:-1,1:-1]')

    # ----------------------------------------------------------------- #
    # computation for kinetic variables and PV (cartesian coordinates)  #
    # ----------------------------------------------------------------- #

    def calc_wind_speed(self):
        if 'u' in dir(self) and 'v' in dir(self):
            self.wind_speed = (self.u**2 + self.v**2)**0.5
        else:
            raise AttributeError('須先設定 u, v')

    def calc_kinetic_energy(self):
        if all(var in dir(self) for var in ['u', 'v', 'w']):
            self.kinetic_energy = \
                (self.u**2 + self.v**2 + self.w**2)/2
        else:
            raise AttributeError('須先設定 u, v, w')

    def calc_vorticity_3D(self):
        if all(var in dir(self) for var in ['u', 'v', 'w']):
            self.zeta_r = FD2(self.w, self.dy, 1) - FD2(self.v, self.dz, 0)
            self.zeta_t = FD2(self.v, self.dz, 0) - FD2(self.w, self.dx, 2)
            self.zeta_z = FD2(self.v, self.dx, 2) - FD2(self.u, self.dy, 1)
        else:
            raise AttributeError('須先設定 u, v, w')

    def calc_abs_vorticity_3D(self):
        if 'zeta_z' not in dir(self):
            self.calc_vorticity_3D()
        if 'f' in dir(self):
            self.abs_zeta_z = self.zeta_z + self.f
        else:
            raise AttributeError('須先設定 f')

    def calc_divergence_hori(self):
        if 'u' in dir(self) and 'v' in dir(self):
            self.div_hori = FD2(self.u, self.dx, 2) \
                + FD2(self.v, self.dy, 1)
        else:
            raise AttributeError('須先設定 u, v')

    def calc_divergence(self):
        if all(var in dir(self) for var in ['u', 'v', 'w']):
            self.div = FD2(self.u, self.dx, 2) \
                + FD2(self.v, self.dy, 1) \
                + FD2(self.w, self.dz, 0)
        else:
            raise AttributeError('須先設定 u, v, w')

    def calc_density_factor(self):
        if 'qv' in dir(self):
            a = 1 + self.qv/0.622
        else:
            raise AttributeError('須先設定 qv')

        if all(var in dir(self) for var in
               ['qc', 'qr', 'qi', 'qs', 'qg', 'qh']):
            b = 1 + self.qv + self.qc + self.qr + self.qi + self.qs + self.qg + self.qh
        elif all(var in dir(self) for var in ['qc', 'qr', 'qi', 'qs', 'qg']):
            b = 1 + self.qv + self.qc + self.qr + self.qi + self.qs + self.qg
        elif all(var in dir(self) for var in ['qc', 'qr', 'qi', 'qs']):
            b = 1 + self.qv + self.qc + self.qr + self.qi + self.qs
        elif all(var in dir(self) for var in ['qc', 'qr', 'qi']):
            b = 1 + self.qv + self.qc + self.qr + self.qi
        elif all(var in dir(self) for var in ['qc', 'qr']):
            b = 1 + self.qv + self.qc + self.qr
        elif all(var in dir(self) for var in ['qc']):
            b = 1 + self.qv + self.qc
        elif all(var in dir(self) for var in ['qr']):
            b = 1 + self.qv + self.qr
        elif all(var in dir(self) for var in []):
            b = 1
            print('Warning: 未設定液態固態水物混和比，計算出的Th_rho將與Th_v相同')
        self.density_factor = a/b

    def calc_density_potential_temperature(self):
        self.calc_density_factor()
        if 'Th' not in dir(self):
            self.calc_theta()   # 須繼承.thermaldynamic_calculation.calc_thermaldynamic
        self.Th_rho = self.Th * self.density_factor

    def calc_PV(self):
        if 'Th_rho' not in dir(self):
            self.calc_density_potential_temperature()
        if 'abs_zeta_z' not in dir(self):
            self.calc_abs_vorticity_3D()
        if 'rho' not in dir(self):
            self.calc_rho()

        r_part = self.zeta_r * FD2(self.Th_rho, self.dx, 2)
        t_part = self.zeta_t * FD2(self.Th_rho, self.dy, 1)
        z_part = self.abs_zeta_z * FD2(self.Th_rho, self.dz, 0)
        self.PV = (r_part + t_part + z_part) / self.rho

    # ----------------------------------------------------------------- #
    # computation for filamentation (cartesian coordinates)             #
    # ----------------------------------------------------------------- #

    def calc_vorticity_z(self):
        if 'u' in dir(self) and 'v' in dir(self):
            self.zeta_z = FD2(self.v, self.dx, 2) - FD2(self.u, self.dy, 1)
        else:
            raise AttributeError('須先設定 u, v')

    def calc_abs_vorticity_z(self):
        if 'zeta_z' not in dir(self):
            self.calc_vorticity_z()
        if 'f' in dir(self):
            self.abs_zeta_z = self.zeta_z + self.f
        else:
            raise AttributeError('須先設定 f')

    def calc_deformation(self):
        if 'u' in dir(self) and 'v' in dir(self):
            self.shear_deform = FD2(self.u, self.dx, 2) \
                - FD2(self.v, self.dy, 1)
            self.stretch_deform = FD2(self.v, self.dx, 2) \
                + FD2(self.u, self.dy, 1)
        else:
            raise AttributeError('須先設定 u, v')

    def calc_filamentation(self):
        if 'shear_deform' not in dir(self) or 'stretch_deform' not in dir(self):
            self.calc_deformation()
        if 'zeta_z' not in dir(self):
            self.calc_vorticity_z()

        self.filamentation_time = 2*(self.shear_deform**2
                                     + self.stretch_deform**2
                                     - self.zeta_z**2)**(-1/2)
        strain_region = \
            self.shear_deform**2 + self.stretch_deform**2 > self.zeta_z**2

        self.filamentation_time = \
            np.where(strain_region, self.filamentation_time, np.inf)

    # ----------------------------------------------------------------- #
    # Average (cartesian coordinates)                                   #
    # ----------------------------------------------------------------- #

    def calc_horizontal_average(self, varname, xmin=np.nan, xmax=np.nan,
                                ymin=np.nan, ymax=np.nan):
        if eval(f'len(self.{varname}.shape)') == 3:
            tmp = calc_Zaverage(eval(f'self.{varname}'), self.x, 2, xmin, xmax)
            exec(
                f'self.havg_{varname} = calc_Zaverage(tmp, self.y, 1, ymin, ymax)')
        elif eval(f'len(self.{varname}.shape)') == 2:  # 假設是(y, x)
            tmp = calc_Zaverage(eval(f'self.{varname}'), self.x, 1, xmin, xmax)
            exec(
                f'self.havg_{varname} = calc_Zaverage(tmp, self.y, 0, ymin, ymax)')
        else:
            raise DimensionError('varname不是3維(z, y, x)或2維(y, x)')

    def calc_vertical_average(self, varname, zmin=np.nan, zmax=np.nan):
        if eval(f'len(self.{varname}.shape)') == 3:
            exec(
                f'self.zavg_{varname} = calc_Zaverage(self.{varname}, self.z, 0, zmin, zmax)')
        elif eval(f'len(self.{varname}.shape)') == 1:  # 假設是(z)
            exec(
                f'self.zavg_{varname} = calc_Zaverage(self.{varname}, self.z, 0, zmin, zmax)')
        else:
            raise DimensionError('varname不是3維(z, y, x)或1維(z)')
