import numpy as np
import multiprocessing as mp
import wrf
from scipy.interpolate import splev, splrep

from ..interpolation import interp2D_fast_layers, interp2D_fast
from ..calc import Make_cyclinder_coord, calc_Tv, calc_rho, FD2
from ..tools import timer
from .grid_general import gridsystem



class cylindrical_grid(gridsystem):
    '''
    圓柱座標網格，3維網格(z, theta, r)，z可為壓力或高度座標，theta為切方向，r為徑方向。
    '''

    def init_grid(self):
        self.meta_convert_to_float('dr', 'dtheta', 'dz', 'dt',
                                   'z_start', 'r_start', 'theta_start')
        self.meta_convert_to_int('Nr', 'Ntheta', 'Nz')

        #依照設定檔建立r、theta、z座標
        self.create_coord()


    def create_coord(self):
        self.r = np.arange(self.r_start, self.Nr*self.dr, self.dr)
        self.theta = np.arange(self.theta_start, self.Ntheta*self.dtheta, self.dtheta)
        self.z = np.arange(0, self.Nz*self.dz, self.dz) + self.z_start

        self.set_coord_meta('z', 'vertical', self.z, unit='m', description='Vertical coordinate')
        self.set_coord_meta('theta', 'tangential', self.theta, unit='rad', description='Tangential coordinate')
        self.set_coord_meta('r', 'radial', self.r, unit='m', description='Radial coordinate')

    def set_horizontal_location(self, offset_x, offset_y):
        '''
        設定內插時的目標位置
        來源資料可能是直角坐標或是wrf座標，設定圓柱座標的水平格點在來源座標的x, y位置，供內插使用
        offset_x, offset_y是來源資料座標系中颱風中心的位置
        '''
        self.xTR, self.yTR = Make_cyclinder_coord([offset_y, offset_x], self.r, self.theta)


    def set_terrain(self,var):
        #決定地形高度，包含地形mask (self.isin_hgt)、地形idx (self.hgt_idx)
        self.hgt_idx = np.sum(np.isnan(var),axis=0) #決定地形idx，統計垂直方向的np.nan加總
        if self.z[0] == 0: #但有時候會在零星幾格單獨存在np.nan 要將這些慮除
            #如果z座標底層高度是0，都會是np.nan，無法識別，所以識別z為第一層，如果這層存在np.nan，表示這裡才有地形，否則給予1 (為第0層的np.nan計數)
            self.hgt_idx = np.where(np.isnan(var[1,:,:]),self.hgt_idx,1)
        else:
            #如果z座標底層高度不是0，此層存在np.nan表示有地形，保留hgt_idx計數，否則將hgt_idx設為0 (np.nan為零星，非地形)
            self.hgt_idx = np.where(np.isnan(var[0,:,:]),self.hgt_idx,0)
        self.isin_hgt = np.where(np.ones([self.Nz,self.Ntheta,self.Nr])*np.arange(self.Nz).reshape(-1,1,1) < self.hgt_idx, True, False)


    def set_data_old(self,wrf_dx,wrf_dy,wrf_vertical,**vardict):
        self.varlist3D = []
        self.varlist2D = []
        for varname, var in vardict.items():
            if len(var.shape) == 3:
                tmp = np.array(wrf.interplevel(var,wrf_vertical,self.z))
                tmp2 = interp2D_fast_layers(wrf_dx, wrf_dy, self.xTR, self.yTR, tmp)
                if self.varlist3D == []:
                    if self.vertical_coord_type == 'z':
                        self.set_terrain(tmp2)
                # tmp2 = self.fill_value(tmp2)
                exec(f'self.{varname} = tmp2')
                self.varlist3D.append(varname)
            elif len(var.shape) == 2:
                exec(f'self.{varname} = interp2D_fast(wrf_dx, wrf_dy, self.xTR, self.yTR, var)')
                self.varlist2D.append(varname)

    def spline(self,var,vert,obj_vert):
            Nx = var.shape[2]
            Ny = var.shape[1]
            out = np.zeros([len(obj_vert),Ny,Nx])

            for j in range(Ny):
                for i in range(Nx):
                    cs = splrep(vert[:,j,i],var[:,j,i],k=2)
                    out[:,j,i] = splev(obj_vert,cs)

            return out

    @timer
    def set_data(self,wrf_dx,wrf_dy,wrf_vertical, ver_interp_order = 1, **vardict):
        self.varlist3D = []
        self.varlist2D = []
        for varname, var in vardict.items():
            if len(var.shape) == 3:
                self.varlist3D.append(varname)
            elif len(var.shape) == 2:
                exec(f'self.{varname} = interp2D_fast(wrf_dx, wrf_dy, self.xTR, self.yTR, var)')
                self.varlist2D.append(varname)

        group = [[vardict[varname],wrf_vertical,self.z] for varname in self.varlist3D]
        if ver_interp_order == 1:
            tmp = []
            for var, vertical, specific_z in group:
                tmp.append(np.array(wrf.interplevel(var, vertical, specific_z)))

                
            '''
            pool = mp.Pool(self.interp_cpus)
            tmp = pool.map(wrf.interplevel,group)
            pool.close()
            pool.join()
            '''
            
        elif ver_interp_order == 2:
            pool = mp.Pool(self.interp_cpus)
            tmp = pool.starmap(self.spline,group)
            pool.close()
            pool.join()
        for idx, varname in enumerate(self.varlist3D):
            #tmp = np.array(wrf.interplevel(var,wrf_vertical,self.z))
            tmp2 = interp2D_fast_layers(wrf_dx, wrf_dy, self.xTR, self.yTR, tmp[idx])
            # tmp2 = self.fill_value(tmp2)
            exec(f'self.{varname} = tmp2')

        if 'hgt' in self.varlist2D:
            self.set_terrain()           #設定地形高度
            self.terrain_mask_vars()     #將地形內部的格點設定為nan


    def set_terrain(self):
        def fix_hgt_idx(hgt_idx):
            hgt_idx[0,1:-1] = np.where(hgt_idx[0,1:-1]<hgt_idx[1,1:-1],hgt_idx[1,1:-1],hgt_idx[0,1:-1])
            hgt_idx[-1,1:-1] = np.where(hgt_idx[-1,1:-1]<hgt_idx[-2,1:-1],hgt_idx[-2,1:-1],hgt_idx[-1,1:-1])
            hgt_idx[1:-1,0] = np.where(hgt_idx[1:-1,0]<hgt_idx[1:-1,1],hgt_idx[1:-1,1],hgt_idx[1:-1,0])
            hgt_idx[1:-1,-1] = np.where(hgt_idx[1:-1,-1]<hgt_idx[1:-1,-2],hgt_idx[1:-1,-2],hgt_idx[1:-1,-1])
            hgt_idx[0,0] = np.max([hgt_idx[0,0],hgt_idx[1,0],hgt_idx[0,1]])
            hgt_idx[0,-1] = np.max([hgt_idx[0,-1],hgt_idx[1,-1],hgt_idx[0,-2]])
            hgt_idx[-1,0] = np.max([hgt_idx[-1,0],hgt_idx[-2,0],hgt_idx[-1,1]])
            hgt_idx[-1,-1] = np.max([hgt_idx[-1,-1],hgt_idx[-2,-1],hgt_idx[-1,-2]])
            return hgt_idx

        #hgt_mask為各網格是否在地形內，以wrfout內的地形高度HGT作為地形高度標準
        #hgt_idx為各水平位置點上，有多少個點位於地形之下
        self.hgt_idx = (self.hgt-self.z_start)//self.dz + 1
        self.hgt_idx = self.hgt_idx.astype('int32')
        self.hgt_idx = np.where(self.hgt_idx<0,0,self.hgt_idx)
        self.hgt_idx = fix_hgt_idx(self.hgt_idx)
        self.hgt_mask = np.where(self.hgt_idx.reshape(1,self.Ntheta,self.Nr) > np.arange(0,self.Nz,1).reshape(-1,1,1),1,0)


    def terrain_mask_vars(self):
        for varname in self.varlist3D:
            exec(f'self.{varname} = np.where(self.hgt_mask,np.nan,self.{varname})')

    def calc_vr_vt(self):
        theta = self.theta.reshape(1,-1,1)
        self.vr = self.v*np.sin(theta) + self.u*np.cos(theta)
        self.vt = self.v*np.cos(theta) - self.u*np.sin(theta)

    def calc_fric_vr_vt(self):
        theta = self.theta.reshape(1,-1,1)
        self.fric_vr = self.fric_v*np.sin(theta) + self.fric_u*np.cos(theta)
        self.fric_vt = self.fric_v*np.cos(theta) - self.fric_u*np.sin(theta)

    def calc_kinetic_energy(self):
        self.kinetic_energy = \
            (self.u**2 + self.v**2 + self.w**2)/2
        
    def calc_Tv(self):
        self.Tv = calc_Tv(self.T, qv=self.qv)
        
    def calc_rho(self):
        self.rho = calc_rho(self.p, T=self.T, qv=self.qv)
        
    def calc_vorticity_3D(self):
        self.zeta_r = FD2(self.w, self.dtheta, 1)/self.r \
                      - FD2(self.vt, self.dz, 0)
        self.zeta_t = FD2(self.vr, self.dz, 0) \
                      - FD2(self.w, self.dr, 2)
        self.zeta_z = FD2(self.vt*self.r, self.dr, 2)/self.r \
                      - FD2(self.vr, self.dtheta, 1)/self.r
        
        
        
    def calc_vorticity_z(self):
        self.zeta_z = FD2(self.vt*self.r, self.dr, 2)/self.r \
                      - FD2(self.vr, self.dtheta, 1)/self.r
    
    def calc_abs_vorticity_3D(self):
        try:
            self.abs_zeta_z = self.zeta_z + self.f
        except AttributeError:
            self.calc_vorticity_3D()
            self.abs_zeta_z = self.zeta_z + self.f
        
    def calc_abs_vorticity_z(self):
        try:
            self.abs_zeta_z = self.zeta_z + self.f
        except AttributeError:
            self.calc_vorticity_z()
            self.abs_zeta_z = self.zeta_z + self.f
            
    def calc_density_factor(self):
        a = 1 + self.qv/0.622
        b = 1 + self.qv + self.qc + self.qr + self.qi + self.qs + self.qg
        self.density_factor = a/b

    def calc_density_potential_temperature(self):
        self.calc_density_factor()
        self.density_Th = self.Th * self.density_factor

    def calc_PV(self):
        r_part = self.zeta_r * FD2(self.density_Th, self.dr, 2)
        t_part = self.zeta_t * FD2(self.density_Th, self.dtheta, 1) / self.r
        z_part = self.abs_zeta_z * FD2(self.density_Th, self.dz, 0)
        self.PV = (r_part + t_part + z_part) / self.rho
