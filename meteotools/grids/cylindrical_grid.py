import numpy as np
import multiprocessing as mp
import wrf
from scipy.interpolate import splev, splrep

from ..interpolation import interp2D_fast_layers, interp2D_fast
from ..calc import Make_cyclinder_coord, FD2, calc_RZaverage, calc_Zaverage, calc_Raverage
from ..tools import timer
from ..exceptions import DimensionError, UnitError
from .grid_general import gridsystem
from .thermaldynamic_calculation import calc_thermaldynamic


class cylindrical_grid(gridsystem, calc_thermaldynamic):
    '''
    圓柱座標網格，3維網格(z, theta, r)，z可為壓力或高度座標，theta為切方向，r為徑方向。
    '''

    def init_grid(self):
        self.meta_convert_to_float('dr', 'dtheta', 'dz', 'dt',
                                   'z_start', 'r_start', 'theta_start')
        self.meta_convert_to_int('Nr', 'Ntheta', 'Nz')

        # 依照設定檔建立r、theta、z座標
        self.create_coord()

    def create_coord(self):
        self.r = np.arange(self.r_start, self.Nr*self.dr, self.dr)
        self.theta = np.arange(
            self.theta_start, self.Ntheta*self.dtheta, self.dtheta)
        self.z = np.arange(0, self.Nz*self.dz, self.dz) + self.z_start

        self.set_coord_meta(
            'z', 'vertical', self.z, unit='m',
            description='Vertical coordinate')
        self.set_coord_meta(
            'theta', 'tangential', self.theta, unit='rad',
            description='Tangential coordinate')
        self.set_coord_meta('r', 'radial', self.r, unit='m',
                            description='Radial coordinate')

    def set_horizontal_location(self, offset_x, offset_y):
        '''
        設定內插時的目標位置
        來源資料可能是直角坐標或是wrf座標，設定圓柱座標的水平格點在來源座標的x, y位置，供內插使用
        offset_x, offset_y是來源資料座標系中颱風中心的位置
        '''
        self.xTR, self.yTR = Make_cyclinder_coord(
            [offset_y, offset_x], self.r, self.theta)


    def set_terrain(self, var):
        # 決定地形高度，包含地形mask (self.isin_hgt)、地形idx (self.hgt_idx)
        self.hgt_idx = np.sum(np.isnan(var), axis=0)  # 決定地形idx，統計垂直方向的np.nan加總
        if self.z[0] == 0:  # 但有時候會在零星幾格單獨存在np.nan 要將這些慮除
            # 如果z座標底層高度是0，都會是np.nan，無法識別，所以識別z為第一層，如果這層存在np.nan，表示這裡才有地形，否則給予1 (為第0層的np.nan計數)
            self.hgt_idx = np.where(np.isnan(var[1, :, :]), self.hgt_idx, 1)
        else:
            # 如果z座標底層高度不是0，此層存在np.nan表示有地形，保留hgt_idx計數，否則將hgt_idx設為0 (np.nan為零星，非地形)
            self.hgt_idx = np.where(np.isnan(var[0, :, :]), self.hgt_idx, 0)
        self.isin_hgt = np.where(np.ones([self.Nz, self.Ntheta, self.Nr])*np.arange(
            self.Nz).reshape(-1, 1, 1) < self.hgt_idx, True, False)

    def set_data_old(self, wrf_dx, wrf_dy, wrf_vertical, **vardict):
        self.varlist3D = []
        self.varlist2D = []
        for varname, var in vardict.items():
            if len(var.shape) == 3:
                tmp = np.array(wrf.interplevel(var, wrf_vertical, self.z))
                tmp2 = interp2D_fast_layers(
                    wrf_dx, wrf_dy, self.xTR, self.yTR, tmp)
                if self.varlist3D == []:
                    if self.vertical_coord_type == 'z':
                        self.set_terrain(tmp2)
                # tmp2 = self.fill_value(tmp2)
                exec(f'self.{varname} = tmp2')
                self.varlist3D.append(varname)
            elif len(var.shape) == 2:
                exec(
                    f'self.{varname} = interp2D_fast(wrf_dx, wrf_dy, self.xTR, self.yTR, var)')
                self.varlist2D.append(varname)

    def spline(self, var, vert, obj_vert):
        Nx = var.shape[2]
        Ny = var.shape[1]
        out = np.zeros([len(obj_vert), Ny, Nx])

        for j in range(Ny):
            for i in range(Nx):
                cs = splrep(vert[:, j, i], var[:, j, i], k=2)
                out[:, j, i] = splev(obj_vert, cs)

        return out

    @timer
    def set_data(
            self, wrf_dx, wrf_dy, wrf_vertical, ver_interp_order=1, **vardict):
        self.varlist3D = []
        self.varlist2D = []
        for varname, var in vardict.items():
            if len(var.shape) == 3:
                self.varlist3D.append(varname)
            elif len(var.shape) == 2:
                exec(
                    f'self.{varname} = interp2D_fast(wrf_dx, wrf_dy, self.xTR, self.yTR, var)')
                self.varlist2D.append(varname)

        group = [[vardict[varname], wrf_vertical, self.z]
                 for varname in self.varlist3D]
        if ver_interp_order == 1:
            tmp = []
            for var, vertical, specific_z in group:
                tmp.append(np.array(wrf.interplevel(
                    var, vertical, specific_z)))

            '''
            pool = mp.Pool(self.interp_cpus)
            tmp = pool.map(wrf.interplevel,group)
            pool.close()
            pool.join()
            '''

        elif ver_interp_order == 2:
            pool = mp.Pool(self.interp_cpus)
            tmp = pool.starmap(self.spline, group)
            pool.close()
            pool.join()
        for idx, varname in enumerate(self.varlist3D):
            #tmp = np.array(wrf.interplevel(var,wrf_vertical,self.z))
            tmp2 = interp2D_fast_layers(
                wrf_dx, wrf_dy, self.xTR, self.yTR, tmp[idx])
            # tmp2 = self.fill_value(tmp2)
            exec(f'self.{varname} = tmp2')

        if 'hgt' in self.varlist2D:
            self.set_terrain()  # 設定地形高度
            self.terrain_mask_vars()  # 將地形內部的格點設定為nan

    def set_terrain(self):
        def fix_hgt_idx(hgt_idx):
            hgt_idx[0, 1: -1] = np.where(hgt_idx[0, 1: -1] <
                                         hgt_idx[1, 1: -1],
                                         hgt_idx[1, 1: -1],
                                         hgt_idx[0, 1: -1])
            hgt_idx[-1, 1: -1] = np.where(hgt_idx[-1, 1: -1] <
                                          hgt_idx[-2, 1: -1],
                                          hgt_idx[-2, 1: -1],
                                          hgt_idx[-1, 1: -1])
            hgt_idx[1: -1, 0] = np.where(hgt_idx[1: -1, 0] <
                                         hgt_idx[1: -1, 1],
                                         hgt_idx[1: -1, 1],
                                         hgt_idx[1: -1, 0])
            hgt_idx[1: -1, -1] = np.where(hgt_idx[1: -1, -1] <
                                          hgt_idx[1: -1, -2],
                                          hgt_idx[1: -1, -2],
                                          hgt_idx[1: -1, -1])
            hgt_idx[0, 0] = np.max(
                [hgt_idx[0, 0], hgt_idx[1, 0], hgt_idx[0, 1]])
            hgt_idx[0, -1] = np.max([hgt_idx[0, -1],
                                    hgt_idx[1, -1], hgt_idx[0, -2]])
            hgt_idx[-1, 0] = np.max([hgt_idx[-1, 0],
                                    hgt_idx[-2, 0], hgt_idx[-1, 1]])
            hgt_idx[-1, -1] = np.max([hgt_idx[-1, -1],
                                     hgt_idx[-2, -1], hgt_idx[-1, -2]])
            return hgt_idx

        # hgt_mask為各網格是否在地形內，以wrfout內的地形高度HGT作為地形高度標準
        # hgt_idx為各水平位置點上，有多少個點位於地形之下
        self.hgt_idx = (self.hgt-self.z_start)//self.dz + 1
        self.hgt_idx = self.hgt_idx.astype('int32')
        self.hgt_idx = np.where(self.hgt_idx < 0, 0, self.hgt_idx)
        self.hgt_idx = fix_hgt_idx(self.hgt_idx)
        self.hgt_mask = np.where(
            self.hgt_idx.reshape(1, self.Ntheta, self.Nr) > np.arange(
                0, self.Nz, 1).reshape(-1, 1, 1),
            1, 0)

    def terrain_mask_vars(self):
        for varname in self.varlist3D:
            exec(
                f'self.{varname} = np.where(self.hgt_mask,np.nan,self.{varname})')

    # ------------------------------------------------------------------- #
    # computation  (cylindrical coordinates only)                         #
    # ------------------------------------------------------------------- #

    def calc_axisymmetry(self,  varname):
        '''
        取指定變數的軸對稱率，指定變數為var，輸出名稱為axisymmetry_var
        若指定變數為3維，假設其維度為(z, theta, r)，輸出2維(z, r)的軸對稱率
        若指定變數為2維，假設其維度為(theta, r)，輸出1維(r)的軸對稱率
        若指定變數為1維，假設其維度為(theta)，輸出單一軸對稱率值
        '''
        if varname not in dir(self):
            raise AttributeError(f'沒有名稱為 {varname} 的變數')

        self.calc_axisymmetric_asymmetric(varname)
        if eval(f'len(self.{varname}.shape)') == 3:  # (z, theta, r)
            exec(
                f'self.axisymmetry_{varname} = self.sym_{varname}**2' +
                f'/(self.sym_{varname}**2 + np.nansum(self.asy_{varname}**2,axis=1)'
                + f'*self.dtheta/2/np.pi)')
        elif eval(f'len(self.{varname}.shape)') == 2:  # (theta, r)
            exec(
                f'self.axisymmetry_{varname} = self.sym_{varname}**2' +
                f'/(self.sym_{varname}**2 + np.nansum(self.asy_{varname}**2,axis=0)'
                + f'*self.dtheta/2/np.pi)')
        elif eval(f'len(self.{varname}.shape)') == 1:  # (theta, r)
            exec(
                f'self.axisymmetry_{varname} = self.sym_{varname}**2' +
                f'/(self.sym_{varname}**2 + np.nansum(self.asy_{varname}**2)'
                + f'*self.dtheta/2/np.pi)')

    def calc_axisymmetric_asymmetric(self, varname):
        '''
        計算指定變數(var)的軸對稱平均(sym_var)與非對稱值(asy_var)
        若指定變數為3維，假設其維度為(z, theta, r)，輸出2維(z, r)的sym_var與
        3維(z, theta, r)的asy_var
        若指定變數為2維，假設其維度為(theta, r)，輸出1維(r)的sym_var率與
        1維(theta, r)的asy_var
        若指定變數為1維，假設其維度為(theta)，輸出單一sym_var值與1維(theta)的asy_var
        '''

        if varname not in dir(self):
            raise AttributeError(f'沒有名稱為 {varname} 的變數')
        if eval(f'len(self.{varname}.shape)') == 3:  # (z, theta, r)
            exec(f'self.sym_{varname} = np.nanmean(self.{varname},axis=1)')
            exec(
                f'self.asy_{varname} = self.{varname} - self.sym_{varname}.reshape(self.Nz, 1, self.Nr)')
        elif eval(f'len(self.{varname}.shape)') == 2:  # (theta, r)
            exec(f'self.sym_{varname} = np.nanmean(self.{varname},axis=0)')
            exec(f'self.asy_{varname} = self.{varname} - self.sym_{varname}')
        elif eval(f'len(self.{varname}.shape)') == 1:  # (theta)
            exec(f'self.sym_{varname} = np.nanmean(self.{varname})')
            exec(f'self.asy_{varname} = self.{varname} - self.sym_{varname}')

    def rotate_to_shear_direction(
            self, varname, direction, unit='rad'):
        '''
        將指定變數(var)依照風切方向旋轉至螢幕正上方為風切方向的陣列(shear_relative_var)
        若指定變數為3維(z, theta, r)，沿第1維(theta)旋轉
        若指定變數為2維(theta, r)，沿第0維(theta)旋轉
        若指定變數為1維(theta)，沿第0維(theta)旋轉
        Parameters
        ----------
        varname : 欲旋轉變數之名稱
        direction : 風切方位角
        unit : 單位，預設為rad，可變更為rad
        '''
        if unit == 'deg':
            direction = np.deg2rad(direction)
        rotate_idx = int(direction/self.dtheta)
        print(rotate_idx)
        if eval(f'len(self.{varname}.shape)') == 3:  # (z, theta, r)
            exec(f'self.shear_relative_{varname} = ' +
                 f'np.roll(self.{varname},{rotate_idx}, axis=1)')
        elif eval(f'len(self.{varname}.shape)') == 2:  # (theta, r)
            exec(f'self.shear_relative_{varname} = ' +
                 f'np.roll(self.{varname},{rotate_idx}, axis=0)')
        elif eval(f'len(self.{varname}.shape)') == 1:  # (theta)
            exec(f'self.shear_relative_{varname} = ' +
                 f'np.roll(self.{varname},{rotate_idx}, axis=0)')

    def calc_vr_vt(self):
        if 'u' in dir(self) and 'v' in dir(self):
            theta = self.theta.reshape(1, -1, 1)
            self.vr = self.v*np.sin(theta) + self.u*np.cos(theta)
            self.vt = self.v*np.cos(theta) - self.u*np.sin(theta)
        else:
            raise AttributeError('須先設定 u, v')

    def calc_vr_vt10(self):
        if 'u10' in dir(self) and 'v10' in dir(self):
            theta = self.theta.reshape(-1, 1)
            self.vr10 = self.v10*np.sin(theta) + self.u10*np.cos(theta)
            self.vt10 = self.v10*np.cos(theta) - self.u10*np.sin(theta)
        else:
            raise AttributeError('須先設定 u10, v10')

    def calc_fric_vr_vt(self):
        if 'fric_u' in dir(self) and 'fric_v' in dir(self):
            theta = self.theta.reshape(1, -1, 1)
            self.fric_vr = self.fric_v*np.sin(
                theta) + self.fric_u*np.cos(theta)
            self.fric_vt = self.fric_v*np.cos(
                theta) - self.fric_u*np.sin(theta)
        else:
            raise AttributeError('須先設定 fric_u, fric_v')

    def calc_aam(self):
        if 'f' not in dir(self):
            raise AttributeError('須先設定f')
        if 'vt' not in dir(self) or 'vr' not in dir(self):
            self.calc_vr_vt()
        r_dim = [1,] * (self.vt.ndim-1) + [-1]
        r = self.r.reshape(r_dim)
        self.aam = self.vt*r + 0.5*self.f*r**2


    def find_rmw_and_speed(self):
        if 'sym_vt' not in dir(self):
            if 'vt' not in dir(self):
                self.calc_vr_vt()
            self.calc_axisymmetric_asymmetric('vt')
        if 'sym_wind_speed' not in dir(self):
            if 'wind_speed' not in dir(self):
                self.calc_wind_speed()
            self.calc_axisymmetric_asymmetric('wind_speed')

        max_wind_r_idx = np.nanargmax(self.sym_wind_speed, axis=1)
        self.RMW = self.r[max_wind_r_idx]
        self.wspd_max_RMW = self.sym_wind_speed[np.arange(
            self.Nz), max_wind_r_idx]
        self.vt_max_RMW = self.sym_vt[np.arange(self.Nz), max_wind_r_idx]


    def find_rmw_and_speed10(self):
        if 'sym_vt10' not in dir(self):
            if 'vt10' not in dir(self):
                self.calc_vr_vt10()
            self.calc_axisymmetric_asymmetric('vt10')
        if 'sym_wind_speed10' not in dir(self):
            if 'wind_speed10' not in dir(self):
                self.calc_wind_speed10()
            self.calc_axisymmetric_asymmetric('wind_speed10')

        max_wind_r_idx = np.nanargmax(self.sym_wind_speed10, axis=1)
        self.RMW10 = self.r[max_wind_r_idx]
        self.wspd_max_RMW10 = self.sym_wind_speed10[max_wind_r_idx]
        self.vt_max_RMW10 = self.sym_vt[max_wind_r_idx]

    # ------------------------------------------------------------------- #
    # computation for agradient force (cylindrical coordinates only)      #
    # ------------------------------------------------------------------- #
    def calc_agradient_force(self):
        if 'vt' not in dir(self):
            self.calc_vr_vt()
        if 'rho' not in dir(self):
            self.calc_rho()
        if 'p' in dir(self) and 'f' in dir(self):
            self.coriolis_force = self.f*self.vt
            self.centrifugal_force = self.vt**2/self.r
            self.radial_PG_force = -FD2(self.p, self.dr, axis=2)/self.rho
            self.agradient_force = self.coriolis_force \
                + self.centrifugal_force \
                + self.radial_PG_force
        else:
            raise AttributeError('須先設定 p, f')

    # ------------------------------------------------------------------- #
    # computation for kinetic variables and PV (cylindrical coordinates)  #
    # ------------------------------------------------------------------- #

    def calc_wind_speed(self):
        if 'u' in dir(self) and 'v' in dir(self):
            self.wind_speed = (self.u**2 + self.v**2)**0.5
        else:
            raise AttributeError('須先設定 u, v')

    def calc_wind_speed10(self):
        if 'u10' in dir(self) and 'v10' in dir(self):
            self.wind_speed10 = (self.u10**2 + self.v10**2)**0.5
        else:
            raise AttributeError('須先設定 u10, v10')

    def calc_kinetic_energy(self):
        if 'vr' not in dir(self) or 'vt' not in dir(self):
            self.calc_vr_vt()
        if 'w' in dir(self):
            self.kinetic_energy = \
                (self.vr**2 + self.vt**2 + self.w**2)/2
        else:
            raise AttributeError('須先設定 w')

    def calc_kinetic_energy10(self):
        if 'vr10' not in dir(self) or 'vt10' not in dir(self):
            self.calc_vr_vt10()
            self.kinetic_energy10 = \
                (self.vr10**2 + self.vt10**2)/2
        else:
            raise AttributeError('須先設定 w')

    def calc_vorticity_3D(self):
        if 'vr' not in dir(self) or 'vt' not in dir(self):
            self.calc_vr_vt()
        if 'w' in dir(self):
            self.zeta_r = FD2(self.w, self.dtheta, 1)/self.r \
                - FD2(self.vt, self.dz, 0)
            self.zeta_t = FD2(self.vr, self.dz, 0) \
                - FD2(self.w, self.dr, 2)
            self.zeta_z = FD2(self.vt*self.r, self.dr, 2)/self.r \
                - FD2(self.vr, self.dtheta, 1)/self.r
        else:
            raise AttributeError('須先設定 w')

    def calc_abs_vorticity_3D(self):
        if 'zeta_z' not in dir(self):
            self.calc_vorticity_3D()
        if 'f' in dir(self):
            self.abs_zeta_z = self.zeta_z + self.f
        else:
            raise AttributeError('須先設定 f')

    def calc_divergence_hori(self):
        if 'vr' not in dir(self) or 'vt' not in dir(self):
            self.calc_vr_vt()

        self.div_hori = FD2(self.r*self.vr, self.dr, 2)/self.r \
            + FD2(self.vt, self.dtheta, 1)/self.r \


    def calc_divergence(self):
        if 'vr' not in dir(self) or 'vt' not in dir(self):
            self.calc_vr_vt()

        if 'w' in dir(self):
            self.div = FD2(self.r*self.vr, self.dr, 2)/self.r \
                + FD2(self.vt, self.dtheta, 1)/self.r \
                + FD2(self.w, self.dz, 0)
        else:
            raise AttributeError('須先設定 w')

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

        r_part = self.zeta_r * FD2(self.Th_rho, self.dr, 2)
        t_part = self.zeta_t * FD2(self.Th_rho,
                                   self.dtheta, 1) / self.r
        z_part = self.abs_zeta_z * FD2(self.Th_rho, self.dz, 0)
        self.PV = (r_part + t_part + z_part) / self.rho

    # ----------------------------------------------------------------- #
    # computation for filamentation (cylindrical coordinates)           #
    # ----------------------------------------------------------------- #

    def calc_vorticity_z(self):
        if 'vr' not in dir(self) or 'vt' not in dir(self):
            self.calc_vr_vt()
        self.zeta_z = FD2(self.vt*self.r, self.dr, 2)/self.r \
            - FD2(self.vr, self.dtheta, 1)/self.r

    def calc_abs_vorticity_z(self):
        if 'zeta_z' not in dir(self):
            self.calc_vorticity_z()
        if 'f' in dir(self):
            self.abs_zeta_z = self.zeta_z + self.f
        else:
            raise AttributeError('須先設定 f')

    def calc_deformation(self):
        if 'vr' not in dir(self) or 'vt' not in dir(self):
            self.calc_vr_vt()
        self.shear_deform = FD2(self.vr, self.dr, 2) - self.vr/self.r \
            - FD2(self.vt, self.dtheta, 1)/self.r
        self.stretch_deform = FD2(self.vt, self.dr, 2) - self.vt/self.r \
            + FD2(self.vr, self.dtheta, 1)/self.r

    def calc_filamentation(self):
        if 'shear_deform' not in dir(self) or 'stretch_deform' not in dir(self):
            self.calc_deformation()
        if 'zeta_z' not in dir(self):
            self.calc_vorticity_z()

        self.filamentation_time = 2*(self.shear_deform**2
                                     + self.stretch_deform**2
                                     - self.zeta_z**2)**(-1/2)
        self.strain_region = \
            self.shear_deform**2 + self.stretch_deform**2 > self.zeta_z**2

        self.filamentation_time = \
            np.where(self.strain_region, self.filamentation_time, np.inf)

    # ----------------------------------------------------------------- #
    # Average (cylindrical coordinates)                                 #
    # ----------------------------------------------------------------- #
    def calc_horizontal_average(self, varname, tmin=np.nan, tmax=np.nan,
                                rmin=np.nan, rmax=np.nan, unit='rad'):
        if unit == 'deg':
            tmin = np.deg2rad(tmin)
            tmax = np.deg2rad(tmax)
        elif unit == 'rad':
            tmin = tmin
            tmax = tmax
        elif unit == 'index':
            tmin = self.theta[tmin]
            tmax = self.theta[tmax]
        else:
            raise UnitError('無效的單位，請輸入deg, rad, index其一。')

        if eval(f'len(self.{varname}.shape)') == 3:
            tmp = calc_Zaverage(eval(f'self.{varname}'),
                                self.theta, 1, tmin, tmax)
            exec(
                f'self.havg_{varname} = calc_Raverage(tmp, self.r, 1, rmin, rmax)')
        elif eval(f'len(self.{varname}.shape)') == 2:  # 假設是(theta, r)
            exec(
                f'self.havg_{varname} = calc_RZaverage(self.{varname}, self.theta, self.r, rmin, rmax, tmin, tmax)')
        else:
            raise DimensionError('varname不是3維(z, theta, r)或2維(theta, r)')

    def calc_vertical_average(self, varname, zmin=np.nan, zmax=np.nan):
        if eval(f'len(self.{varname}.shape)') == 3:
            exec(
                f'self.zavg_{varname} = calc_Zaverage(self.{varname}, self.z, 0, zmin, zmax)')
        elif eval(f'len(self.{varname}.shape)') == 1:  # 假設是(z)
            exec(
                f'self.zavg_{varname} = calc_Zaverage(self.{varname}, self.z, 0, zmin, zmax)')
        else:
            raise DimensionError('varname不是3維(z, theta, r)或1維(z)')

    def create_filter_axis(self, axis, vmin, vmax):
        faxis = np.ones(axis.shape)
        faxis = np.where(axis >= vmin, faxis, 0)
        faxis = np.where(axis <= vmax, faxis, 0)
        return faxis

    def calc_IKE(self, rmin=np.nan, rmax=np.nan, zmin=np.nan, zmax=np.nan):
        if 'kinetic_energy' not in dir(self):
            self.calc_kinetic_energy()
        if 'rho' not in dir(self):
            self.calc_rho()
        
        if np.isnan(rmin):
            rmin = self.r[0]
        if np.isnan(rmax):
            rmax = self.r[-1]
        if np.isnan(zmin):
            zmin = self.z[0]
        if np.isnan(zmax):
            zmax = self.z[-1]
            
        filter_r = self.create_filter_axis(self.r, rmin, rmax)
        filter_z = self.create_filter_axis(self.z, zmin, zmax)
        r3d = np.stack([np.stack([self.r]*self.Ntheta)]*self.Nz)
        volume = r3d*self.dr*self.dtheta*self.dz
        volume *= filter_z.reshape(-1, 1, 1)
        volume *= filter_r.reshape(1, 1, -1)
        self.IKE = np.nansum(self.rho*self.kinetic_energy*volume)

    def calc_IKE10(self, rmax = np.nan, min_wspd10 = 0., rho = np.nan):
        if 'kinetic_energy10' not in dir(self):
            self.calc_kinetic_energy10()
        if 'wind_speed10' not in dir(self):
            self.calc_wind_speed10()
        if 'rho' not in dir(self):
            self.calc_rho()
            zidx = np.nanargmin(np.abs(self.z - 10.))
            if np.isnan(rho):
                rho = self.rho[zidx,:,:]
            else:
                rho = self.rho[zidx,:,:]*0 + rho

        r2d = np.stack([self.r]*self.Ntheta)
        if not (np.isnan(rmax)):
            r2d = np.where(r2d <= rmax, r2d, 0.)
        r2d = np.where(self.wind_speed10 > min_wspd10, r2d, 0.)
        h = 1.
        volume = r2d*self.dr*self.dtheta*h
        self.IKE10 = np.nansum(rho*self.kinetic_energy10*volume)

