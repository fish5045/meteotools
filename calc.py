import numpy as np
import pyproj as pj
from scipy.fft import fft, ifft, fftfreq

if __name__ == 'meteotools.calc':
    from .exceptions import InputError
else:
    from exceptions import InputError

#熱力參數
Rd = 287. # J/kg*K
Cp = 1004. # J/kg*K
g = 9.8 # m/s^2
T0 = 250. # K
P0 = 1000. # hPa
H = Rd*T0/g # m
kappa = Rd/Cp
Lv = 2.5e6
A=2.53e11
B=5420.

def calc_RZaverage(var, z_axis, r_axis, rmin=np.nan, rmax=np.nan, zmin=np.nan, zmax=np.nan, r_weight=True):
    if np.min((z_axis[1:]-z_axis[:-1])<0):
        raise ValueError("z axis must increase strictly.")
    if np.min((r_axis[1:]-r_axis[:-1])<0):
        raise ValueError("r axis must increase strictly.")
    if np.isnan(rmin):
        rmin = r_axis[0]
    if np.isnan(rmax):
        rmax = r_axis[-1]
    if np.isnan(zmin):
        zmin = z_axis[0]
    if np.isnan(zmax):
        zmax = z_axis[-1]
    
    z_select = np.logical_and(z_axis>=zmin, z_axis<=zmax)
    r_select = np.logical_and(r_axis>=rmin, r_axis<=rmax)
    
    if r_weight==True:
        varout = np.nansum(var*r_select*r_axis,axis=1)/np.nansum(r_select*r_axis)
    else:
        varout = np.nanmean(var[:,r_select],axis=1)
    varout = np.nanmean(varout[z_select])
    return varout


def calc_Raverage(var, r_axis, axis, rmin=np.nan, rmax=np.nan):
    if np.min((r_axis[1:]-r_axis[:-1])<=0):
        raise ValueError("r axis must increase strictly.")
    if np.isnan(rmin):
        rmin = r_axis[0]
    if np.isnan(rmax):
        rmax = r_axis[-1]
    
    r_select = np.logical_and(r_axis>=rmin, r_axis<=rmax)

    varout = np.nansum(np.moveaxis(var,axis,-1)*r_select*r_axis,axis=-1)/np.nansum(r_select*r_axis)
    return varout


def calc_Zaverage(var, z_axis, axis, zmin=np.nan, zmax=np.nan):
    if np.min((z_axis[1:]-z_axis[:-1])<=0):
        raise ValueError("z axis must increase strictly.")
    if np.isnan(zmin):
        zmin = z_axis[0]
    if np.isnan(zmax):
        zmax = z_axis[-1]
    
    z_select = np.logical_and(z_axis>=zmin, z_axis<=zmax)
    varout = np.nanmean(np.moveaxis(var,axis,0)[z_select],axis=0)
    return varout

def difference_FFT(var,delta,axis):
    var = np.moveaxis(var, axis,-1)
    yf = fft(var)*2*np.pi/delta
    n = yf.shape[-1]
    yf *= 1j*fftfreq(n)
    output = ifft(yf)
    output = np.moveaxis(output, axis,-1)
    return output


def FD2(var,delta,axis,cyclic=False):
    '''
    以二階中差分法（邊界使用二階偏差分法）計算變數var的微分值。

    Parameters
    ----------
    var : array
        要計算差分的陣列。
    delta : float
        此維度的網格間距。
    axis : int
        要取微分的維度。
    cyclic : bool, optional
        此方向是否是循環而週期性的（如圓柱座標的切方向）
        預設為False。

    Returns
    -------
    output : array (dim same as var)
        二階差分法微分值。
    '''
    
    delta = float(delta)
    
    # 此方向為週期性
    if cyclic == True:
        output = (np.roll(var,-1,axis)-np.roll(var,1,axis))/2/delta

    # 非週期性
    elif cyclic == False:
        var = np.moveaxis(var, axis,0)
        output = np.zeros(var.shape)*np.nan
        output[0] = (-3*var[0]+4*var[1]-var[2])/2/delta
        output[1:-1] = (var[2:]-var[:-2])/2/delta
        output[-1] = (3*var[-1]-4*var[-2]+var[-3])/2/delta
        output = np.moveaxis(output,0,axis)


    return output

def FD2_front(var,delta,axis):
    '''
    以二階前差分法計算變數var的微分值（下邊界兩排為nan）。

    Parameters
    ----------
    var : array
        要計算差分的陣列。
    delta : float
        此維度的網格間距。
    axis : int
        要取微分的維度。

    Returns
    -------
    output : array (dim same as var)
        二階前差分法微分值。
    '''
    
    delta = float(delta)
    var = np.moveaxis(var, axis,0)
    output = np.zeros(var.shape)*np.nan
    output[:-2] = (-3*var[:-2]+4*var[1:-1]-var[2:])/2/delta
    output = np.moveaxis(output,0,axis)
    return output


def FD2_back(var,delta,axis):
    '''
    以二階後差分法計算變數var的微分值（上邊界兩排為nan）。

    Parameters
    ----------
    var : array
        要計算差分的陣列。
    delta : float
        此維度的網格間距。
    axis : int
        要取微分的維度。

    Returns
    -------
    output : array (dim same as var)
        二階後差分法微分值。
    '''
    delta = float(delta)
    var = np.moveaxis(var, axis,0)
    output = np.zeros(var.shape)*np.nan
    output[2:] = (3*var[2:]-4*var[1:-1]+var[:-2])/2/delta
    output = np.moveaxis(output,0,axis)

    return output

def FD2_2(var,delta,axis):
    '''
    以二階中差分法計算變數var的二次微分值（邊界點為偏差分法，準確度二階）。

    Parameters
    ----------
    var : array
        要計算差分的陣列。
    delta : float
        此維度的網格間距。
    axis : int
        要取二次微分的維度。

    Returns
    -------
    output : array (dim same as var)
        二階中差分法二次微分值。
    '''
    
    delta = float(delta)
    var = np.moveaxis(var, axis,0)
    output = np.zeros(var.shape)*np.nan
    output[0] = (2*var[0]-5*var[1]+4*var[2]-var[3])/delta/delta
    output[1:-1] = (var[2:]-2*var[1:-1]+var[:-2])/delta/delta
    output[-1] = (2*var[-1]-5*var[-2]+4*var[-3]-var[-4])/delta/delta
    output = np.moveaxis(output,0,axis)
    return output

def FD2_2_front(var,delta,axis):
    '''
    以二階前差分法計算變數var的二次微分值（下邊界兩排為nan）。

    Parameters
    ----------
    var : array
        要計算差分的陣列。
    delta : float
        此維度的網格間距。
    axis : int
        要取二次微分的維度。

    Returns
    -------
    output : array (dim same as var)
        二階前差分法二次微分值。
    '''
    
    delta = float(delta)
    var = np.moveaxis(var, axis,0)
    output = np.zeros(var.shape)*np.nan
    output[:-3] = (2*var[:-3]-5*var[1:-2]+4*var[2:-1]-var[3:])/delta/delta
    output = np.moveaxis(output,0,axis)
    return output

def FD2_2_back(var,delta,axis):
    '''
    以二階後差分法計算變數var的二次微分值（上邊界兩排為nan）。

    Parameters
    ----------
    var : array
        要計算差分的陣列。
    delta : float
        此維度的網格間距。
    axis : int
        要取二次微分的維度。

    Returns
    -------
    output : array (dim same as var)
        二階後差分法二次微分值。
    '''

    delta = float(delta)
    var = np.moveaxis(var, axis,0)
    output = np.zeros(var.shape)*np.nan
    output[3:] = (2*var[3:]-5*var[2:-1]+4*var[1:-2]-var[:-3])/delta/delta
    output = np.moveaxis(output,0,axis)
    return output


def NanWeightAvg(a,r):
    '''
    忽略無效值的加權平均，可用於圓柱座標徑向平均。

    Parameters
    ----------
    a : array
        欲取平均的資料。
    r : array (dim same as a)
        權重。

    Returns
    -------
    s : array
        加權平均結果
    '''

    rr = np.copy(r)
    s = 0
    for i in range(len(a)):
        if np.isnan(a[i]) or np.isinf(a[i]):
            rr[i] = 0
        else:
            s += a[i]*rr[i]
    s/= np.sum(rr)
    return s


def Find_Root2(a,b,c):
    '''
    求解 ax^2 + bx + c = 0 solve x 的根

    Parameters
    ----------
    a : float or array
        二次項係數
    b : float or array
        一次項係數
    c : float or array
        常數項

    Returns
    -------
    x1 : float or array (complex if b^2-4ac < 0)
        正根
    x2 : float or array (complex if b^2-4ac < 0)
        負根

    '''
    x1 = (-b+(b**2-4*a*c)**0.5)/(2*a)
    x2 = (-b-(b**2-4*a*c)**0.5)/(2*a)
    return x1, x2


def wswd2uv(ws, wd):
    '''
    給入風速風向，轉為東西、南北風

    Parameters
    ----------
    ws : float or array
        風速
    wd : float or array
        風向

    Returns
    -------
    u : float or array 
        東西風
    v : float or array
        南北風

    '''
    
    u = -np.sin(wd / 180. * np.pi) * ws
    v = -np.cos(wd / 180. * np.pi) * ws
    return u,v


def uv2wswd(u, v):
    '''
    給入向東西、南北風，轉為風速風

    Parameters
    ----------
    u : float or array
        東西風
    v : float or array
        南北風

    Returns
    -------
    ws : float or array 
        風速
    wd : float or array
        風向

    '''
    
    ws = np.sqrt((u * u) + (v * v))
    wd = (np.arctan2(-u, -v) * 180. / np.pi) % 360
    return ws, wd


def xy2lonlat(xx, yy, proj='lcc', lat_1=22., lat_2=25., lat_0=23.5, lon_0=120., x_0=0., y_0=0., a=6378137., rf=298.257222101, to_meter=1.):
    '''
    地圖xy座標轉為lon lat
    輸入: 地圖x座標             xx            float or array
         地圖y座標             yy            float or array
    輸出: 經度lon              lon           float or array
         緯度lat              lat           float or array
    選項: 各地圖資訊            proj, lat_1, lat_2, lat_0, lon_0, x_0, y_0, a, rf, to_meter
    '''
    
    isn2004=pj.Proj(f"+proj={proj} +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat_0} +lon_0={lon_0} +x_0={x_0} +y_0={y_0} +no_defs +a={a} +rf={rf} +to_meter={to_meter}")
    return isn2004(xx,yy,inverse=True)


def lonlat2xy(lon, lat, proj='lcc', lat_1=22., lat_2=25., lat_0=23.5, lon_0=120., x_0=0., y_0=0., a=6378137., rf=298.257222101, to_meter=1.):
    '''
    地圖xy座標轉為lon lat
    輸出:  經度lon              lon           float or array
          緯度lat              lat           float or array
    輸入: 地圖x座標             xx            float or array
         地圖y座標             yy            float or array
    選項: 各地圖資訊            proj, lat_1, lat_2, lat_0, lon_0, x_0, y_0, a, rf, to_meter
    '''
    
    isn2004=pj.Proj(f"+proj={proj} +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat_0} +lon_0={lon_0} +x_0={x_0} +y_0={y_0} +no_defs +a={a} +rf={rf} +to_meter={to_meter}")
    return isn2004(lon, lat,inverse=False)

def calc_H(Tv):
    '''
    計算scale height (壓力於平均虛溫大氣中下降至1/e所經的垂直距離)

    Parameters
    ----------
    Tv : float or array
        平均虛溫(K)

    Returns
    -------
    float or array
        scale height (m)

    '''

    return Rd*Tv/g


def PtoZ(P0,Tv,P):
    '''
    將壓力轉為高度，假設溫度相同(溫度會影響到厚度)

    Parameters
    ----------
    P0 : float or array
        參考壓力 (Pa)
    Tv : float or array
        平均虛溫 (K)
    P : float or array
        指定壓力 (Pa)

    Returns
    -------
    float or array
        P0上升至P的垂直距離 (m)
    '''

    H = calc_H(Tv)
    return np.log(P0/P)*H

def ZtoP(P0,Tv,Z):
    '''
    將高度轉為壓力，假設溫度相同(溫度會影響到厚度)

    Parameters
    ----------
    P0 : float or array
        參考壓力 (Pa)
    Tv : float or array
        平均虛溫 (K)
    Z : float or array
        指定高度 (m)

    Returns
    -------
    float or array
        由參考壓力P0，上升Z公尺高後的壓力
    '''

    H = calc_H(Tv)
    return P0*np.exp(-Z/H)

def calc_theta(T, P):
    '''
    計算位溫

    Parameters
    ----------
    T : float or array
        溫度 (K)
    P : float or array
        壓力 (Pa)

    Returns
    -------
    float or array
        位溫 (K)
    '''
    return T*(100000/P)**kappa


def calc_theta_v(T=None, P=None, theta=None, qv=None, es=None, RH=None, Td=None, ql=0):
    '''
    

    Parameters
    ----------
    T : float or array, optional
        溫度 (K) The default is None.
    P : float or array, optional
        壓力 (Pa) The default is None.
    theta : float or array, optional
        位溫 (K) The default is None.
    qv : float or array, optional
        混和比 (kg/kg) The default is None.
    es : float or array, optional
        水氣壓 (Pa) The default is None.
    RH : float or array, optional
        相對溼度 The default is None.
    Td : float or array, optional
        露點溫度 (K) The default is None.
    ql : float or array, optional
        液態水含量 (kg/kg) The default is 0.

    Raises
    ------
    InputError
        輸入資訊不足，詳見錯誤訊息

    Returns
    -------
    float or array
        虛位溫 (K)

    '''
    if type(theta) == type(None):
        if type(T) != type(None) and type(P) != type(None):    
            theta = T*(100000/P)**kappa
        else:
            raise InputError('theta 或 (T,P) 須至少輸入其中一者')
    if type(qv) == type(None):
        if type(es) != type(None) and type(P) != type(None):
            qv = calc_qv(P=P, vapor=es)
        elif type(T) != type(None) and type(RH) != type(None) and type(P) != type(None):
            qv = calc_qv(P=P,T=T,RH=RH)
        elif type(Td) != type(None) and type(P) != type(None):
            qv = calc_qv(P=P,Td=Td)
        else: 
            raise InputError('(vapor, P) 或 (T,RH,P) 或 (Td,P) 須至少輸入其中一者')
    return theta*(1+qv*0.61-ql)


def calc_T(theta,P):
    '''
    以位溫計算溫度

    Parameters
    ----------
    theta : float or array
        位溫 (K)
    P : float or array
        壓力 (Pa)

    Returns
    -------
    float or array
        溫度 (K)
    '''

    return theta*(P/100000)**kappa


def calc_saturated_vapor(T):
    '''
    計算飽和水氣壓

    Parameters
    ----------
    T : float or array
        溫度 (K)

    Returns
    -------
    float or array
        水氣壓 (Pa)
    '''
    return 611.2*np.exp(17.67*(T-273.15)/(T-273.15+243.5))

def calc_vapor(T, RH=1):
    '''
    計算水氣壓

    Parameters
    ----------
    T : float or array
        溫度 (K)
    RH : float or array, optional
        相對溼度，預設為1，此時T為Td

    Returns
    -------
    float or array
        水氣壓 (Pa)
        
    '''
    
    return RH*calc_saturated_vapor(T)
    
def calc_saturated_qv(T, P):
    '''
    計算飽和混和比

    Parameters
    ----------
    T : float or array
        溫度 (K)
    P : float or array
        壓力 (Pa)

    Returns
    -------
    float or array
        飽和混和比 (kg/kg)
    '''
    return 0.622*calc_saturated_vapor(T)/(P-calc_saturated_vapor(T))

def calc_qv(P, T=None, RH=None, vapor=None, Td=None):
    '''
    計算混和比

    Parameters
    ----------
    P : float or array
        壓力 (Pa)
    T : float or array, optional
        溫度 (K) The default is None.
    RH : float or array, optional
        相對濕度 The default is None.
    vapor : float or array, optional
        水氣壓 (Pa) The default is None.
    Td : float or array, optional
        露點溫度 (K) The default is None.

    Returns
    -------
    float or array
        混和比 (kg/kg)

    '''
    if type(vapor) == type(None):
        if type(RH) != type(None) and type(T) != type(None):
            vapor = calc_vapor(T, RH)
        elif type(Td) != type(None):
            vapor = calc_vapor(Td)
        else:
            raise InputError('(T,RH) 、 Td 或 vapor須至少輸入一者')
    return 0.622*vapor/(P-vapor)


def qv2vapor(P, qv):
    '''
    

    Parameters
    ----------
    P : float or array
        壓力 (Pa)
    qv : float or array
        混和比 (kg/kg)

    Returns
    -------
    float or array
        水氣壓 (Pa)

    '''
    return P*qv/(0.622+qv)

def calc_Tv(T, P=None, RH=None, vapor=None, qv=None):
    '''
    計算飽和混和比

    Parameters
    ----------
    T : float or array,
        溫度 (K) 
    P : float or array, optional
        壓力 (Pa) The default is None.
    RH : float or array, optional
        相對濕度 The default is None.
    vapor : float or array, optional
        水氣壓 (Pa) The default is None.
    qv : float or array, optional
        混和比 (kg,kg) The default is None.

    Returns
    -------
    float or array
        虛溫 (K)
    '''
    
    if type(RH) != type(None):
        return T/(1-(calc_vapor(T, RH)/P)*(1-0.622))
    elif type(vapor) != type(None):
        return T/(1-(vapor/P)*(1-0.622))
    elif type(qv) != type(None):
        return T*(1 + (qv/0.622)) / (1 + qv)
    else:
        raise InputError('RH 或 vapor 或 qv須至少輸入一者')


def calc_rho(P, Tv=None, T=None, RH=None, vapor=None, qv=None):
    '''
    

    Parameters
    ----------
    P : float or array, optional
        壓力 (Pa) The default is None.
    Tv : float or array
        虛溫 (K) The default is None.
    T : float or array, optional
        溫度 (K) The default is None.
    RH : float or array, optional
        相對濕度 The default is None.
    vapor : float or array, optional
        水氣壓 (Pa) The default is None.
    qv : float or array, optional
        混和比 (kg,kg) The default is None.

    Returns
    -------
    float or array
        空氣密度 (kg/m3)
    '''
    if type(Tv) == type(None):
        Tv = calc_Tv(T, P=P, RH=RH, vapor=vapor, qv=qv)

    return P/(Rd*Tv)



'''
可能有問題
'''
def calc_Te(T, P=None, RH=None, vapor=None, qv=None):
    '''
    計算相當溫度

    Parameters
    ----------
    T : float or array
        溫度 (K)
    P : float or array, optional
        壓力 (Pa) The default is None.
    RH : float or array, optional
        相對濕度 The default is None.
    vapor : float or array, optional
        水氣壓 (Pa) The default is None.
    qv : float or array, optional
        混和比 (kg,kg) The default is None.

    Returns
    -------
    float or array
        相當溫度 (K)
    '''
    if type(RH) != type(None) and type(P) != type(None):
        return T + Lv/Cp*calc_qv(P, T, RH)
    elif type(vapor) != type(None) and type(P) != type(None):
        return T + Lv/Cp*calc_qv(P, vapor)
    elif type(qv) != type(None):
        return T + Lv/Cp*qv
    else:
        raise InputError('(P,RH) 或 (P,vapor) 或 qv須至少輸入一者')

def calc_theta_e2(P, T=None, RH=None, vapor=None, qv=None, Te=None):
    '''
    計算相當位溫

    Parameters
    ----------
    P : float or array
        溫度 (K)
    T : float or array, optional
        壓力 (Pa) The default is None.
    RH : float or array, optional
        相對濕度 The default is None.
    vapor : float or array, optional
        水氣壓 (Pa) The default is None.
    qv : float or array, optional
        混和比 (kg,kg) The default is None.
    Te : float or array, optional
        相當溫度 (kg,kg) The default is None.

    Returns
    -------
    float or array
        相當位溫 (K)
    '''
    if type(RH) != type(None) and type(P) != type(None):
        return calc_Te(T, P, RH=RH)*(100000/P)**(Rd/Cp)
    elif type(vapor) != type(None) and type(P) != type(None):
        return calc_Te(T, P, vapor=vapor)*(100000/P)**(Rd/Cp)
    elif type(qv) != type(None):
        return calc_Te(T, qv=qv)*(100000/P)**(Rd/Cp)
    elif type(Te) != type(None):
        return Te*(100000/P)**(Rd/Cp)
    else:
        raise InputError('(P,RH) 或 (P,vapor) 或 qv 或 Te須至少輸入一者')

def calc_Tc(T, P=None, RH=None, vapor=None, qv=None):
    '''
    計算凝結溫度condensation temperature

    Parameters
    ----------
    T : float or array
        溫度 (Pa)
    P : float or array, optional
        壓力 (K) The default is None.
    RH : float or array, optional
        相對濕度 The default is None.
    vapor : float or array, optional
        水氣壓 (Pa) The default is None.
    qv : float or array, optional
        混和比 (kg,kg) The default is None.

    Returns
    -------
    float or array
        凝結溫度 (K)
    '''
    if  type(qv) == type(None):
        if type(RH) != type(None) and type(P) != type(None):
            qv = calc_qv(P=P, T=T, RH=RH)
        elif type(vapor) != type(None) and type(P) != type(None):
            qv = calc_qv(P=P, vapor=vapor)
        else:
            raise InputError('qv 或 (RH,P) 或 (vapor,P)須至少輸入一者')
    
    Tc2 = T
    Tc = 0
    for i in range(10):
        Tc = B/(np.log(A*0.622/P/qv*(T/Tc2)**(Cp/Rd)))
        if np.max(np.abs(Tc-Tc2)) < 0.1:
            break
        Tc2 = Tc
    return Tc

    
def calc_theta_e(T, theta=None, P=None, RH=None, vapor=None, qv=None):
    '''
    計算相當位溫

    Parameters
    ----------
    T : float or array
        溫度 (Pa)
    theta : float or array, optional
        位溫 (K) The default is None.
    P : float or array, optional
        壓力 (K) The default is None.
    RH : float or array, optional
        相對濕度 The default is None.
    vapor : float or array, optional
        水氣壓 (Pa) The default is None.
    qv : float or array, optional
        混和比 (kg,kg) The default is None.

    Returns
    -------
    float or array
        相當位溫 (K)
    '''
    if type(theta) == type(None):
        if type(P) != type(None):
            theta = calc_theta(T, P)
        else:
            raise InputError('theta 或 P須至少輸入一者')
    if type(qv) == type(None):
        if type(RH) != type(None) and type(P) != type(None):
            qv = calc_qv(P, T, RH=RH)
        elif type(vapor) != type(None) and type(P) != type(None):
            qv = calc_qv(P, vapor=vapor)
        else:
            raise InputError('qv 或 (RH,P) 或 (vapor,P)須至少輸入一者')
    if type(P) != type(None):
        Tc = calc_Tc(T,P,qv=qv)
    else:
        raise InputError('需輸入P')
    return theta*np.exp(Lv*qv/Cp/Tc)

def calc_theta_es(T, P, theta=None):
    '''
    計算相當位溫

    Parameters
    ----------
    T : float or array
        溫度 (Pa)
    P : float or array
        壓力 (K) 
    theta : float or array, optional
        位溫 (K) The default is None.

    Returns
    -------
    float or array
        相當位溫 (K)
    '''
    if type(theta) == type(None):
            theta = calc_theta(T, P)
    
    qv = calc_saturated_qv(T, P)
    
    return theta*np.exp(Lv*qv/Cp/T)

def calc_dTdz(saturated,T,qv):
    '''
    計算絕熱溫度遞減率

    Parameters
    ----------
    saturated : bool
        氣塊是否飽和
    T : float or array
        溫度 (Pa)
    qv : float or array
        混和比 (kg/kg) 

    Returns
    -------
    float or array
        絕熱溫度遞減率 (K/m)
    '''

    if saturated == False:
        return -g/Cp
    else:
        return -g*((1 + Lv*qv/Rd/T) / (Cp + Lv**2*qv*0.622/Rd/T**2))

def calc_Td(es=None,P=None,qv=None):
    '''
    計算露點溫度

    Parameters
    ----------
    es : float or array, optional
        水氣壓 (Pa) The default is None.
    P : float or array, optional
        氣壓 (Pa) The default is None.
    qv : float or array, optional
        混和比 (kg/kg) The default is None.

    Returns
    -------
    float or array
        露點溫度 (K)

    '''

    if type(qv) != type(None) and type(P) != type(None):
        es = qv2vapor(P, qv)
    elif type(es) == type(None):
        raise InputError('需輸入es 或 (P, qv)')   
        
    lnes = np.log(es/611.2)
    return 243.5*lnes / (17.67-lnes) + 273.15
