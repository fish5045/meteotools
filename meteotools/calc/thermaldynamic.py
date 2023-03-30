import numpy as np
from meteotools.exceptions import InputError

Rd = 287.  # J/kg*K
Cp = 1004.  # J/kg*K
g = 9.8  # m/s^2
T0 = 250.  # K
P0 = 1000.  # hPa
H = Rd*T0/g  # m
kappa = Rd/Cp
Lv = 2.5e6
A = 2.53e11
B = 5420.


def calc_H(Tv):
    '''
    計算scale height (壓力於平均虛溫大氣中下降至1/e所經的垂直距離)
    可為float或陣列
    Parameters
    ----------
    Tv : 平均虛溫(K)

    Returns
    -------
    scale height (m)
    '''

    return Rd*Tv/g


def P_to_Z(P0, Tv, P):
    '''
    將壓力轉為高度，假設溫度相同(溫度會影響到厚度)
    可為float或陣列
    Parameters
    ----------
    P0 : 參考壓力 (Pa)
    Tv : 平均虛溫 (K)
    P : 指定壓力 (Pa)

    Returns
    -------
    P0上升至P的垂直距離 (m)
    '''

    H = calc_H(Tv)
    return np.log(P0/P)*H


def Z_to_P(P0, Tv, Z):
    '''
    將高度轉為壓力，假設溫度相同(溫度會影響到厚度)
    可為float或陣列
    Parameters
    ----------
    P0 : 參考壓力 (Pa)
    Tv : 平均虛溫 (K)
    Z : 指定高度 (m)

    Returns
    -------
    由參考壓力P0，上升Z公尺高後的壓力
    '''

    H = calc_H(Tv)
    return P0*np.exp(-Z/H)


def calc_theta(T, P):
    '''
    計算位溫
    可為float或陣列
    Parameters
    ----------
    T : 溫度 (K)
    P : 壓力 (Pa)

    Returns
    -------
    位溫 (K)
    '''

    return T*(100000/P)**kappa


def calc_T(theta, P):
    '''
    以位溫計算溫度

    Parameters
    ----------
    theta : 位溫 (K)
    P : 壓力 (Pa)

    Returns
    -------
    溫度 (K)
    '''

    return theta*(P/100000)**kappa


def calc_saturated_vapor(T):
    '''
    計算飽和水氣壓

    Parameters
    ----------
    T : 溫度 (K)

    Returns
    -------
    水氣壓 (Pa)
    '''
    return 611.2*np.exp(17.67*(T-273.15)/(T-273.15+243.5))


def calc_vapor(T, RH=1.):
    '''
    計算水氣壓

    Parameters
    ----------
    T : 溫度 (K)
    RH : 相對溼度，預設為1，此時T為Td

    Returns
    -------
    水氣壓 (Pa)

    '''

    return RH*calc_saturated_vapor(T)


def calc_saturated_qv(T, P):
    '''
    計算飽和混和比

    Parameters
    ----------
    T : 溫度 (K)
    P : 壓力 (Pa)

    Returns
    -------
    飽和混和比 (kg/kg)
    '''

    return 0.622*calc_saturated_vapor(T)/(P-calc_saturated_vapor(T))


def qv_to_vapor(P, qv):
    '''
    混和比轉換至水氣壓

    Parameters
    ----------
    P : 壓力 (Pa)
    qv : 混和比 (kg/kg)

    Returns
    -------
    水氣壓 (Pa)
    '''

    return P*qv/(0.622+qv)


def calc_theta_es(T, P):
    '''
    計算相當位溫

    Parameters
    ----------
    T : 溫度 (Pa)
    P : 壓力 (K)

    Returns
    -------
    float or array
        相當位溫 (K)
    '''

    theta = calc_theta(T, P)
    qv = calc_saturated_qv(T, P)

    return theta*np.exp(Lv*qv/Cp/T)


def calc_Td(es=None, P=None, qv=None, T=None, RH=None, theta=None):
    '''
    計算露點溫度
    選擇性輸入必要資訊，es, (P, qv), (T, RH), 或(theta, P, RH)至少輸入其中之一。
    Parameters
    ----------
    es : 水氣壓 (Pa)
    P : 氣壓 (Pa)
    qv : 混和比 (kg/kg)
    T : 溫度 (K)
    RH : 相對濕度
    theta : 位溫 (K)

    Returns
    -------
    露點溫度 (K)

    '''
    def check_es(es):
        if type(es) is type(None):
            if type(None) not in [type(qv), type(P)]:
                return qv_to_vapor(P, qv)
            elif type(None) not in [type(T), type(RH)]:
                return calc_vapor(T, RH)
            elif type(None) not in [type(theta), type(P), type(RH)]:
                T = calc_T(theta, P)
                return calc_vapor(T, RH)
            else:
                raise InputError('需輸入es, (P, qv), (T, RH), 或(theta, P, RH)')
        return es

    es = check_es(es)
    lnes = np.log(es/611.2)
    return 243.5*lnes / (17.67-lnes) + 273.15


def calc_qv(P, T=None, RH=None, vapor=None, Td=None):
    '''
    計算混和比
    選擇性輸入必要資訊，(T,RH) 或 Td 或 vapor至少輸入其中之一。
    Parameters
    ----------
    P : 壓力 (Pa)
    T : 溫度 (K)
    RH : 相對濕度
    vapor : 水氣壓 (Pa)
    Td : 露點溫度 (K)

    Returns
    -------
    混和比 (kg/kg)

    '''
    def check_vapor(vapor):
        if type(vapor) is type(None):
            if type(None) not in [type(RH), type(T)]:
                vapor = calc_vapor(T, RH)
            elif type(None) not in [type(Td)]:
                vapor = calc_vapor(Td)
            else:
                raise InputError('(T,RH) 、 Td 或 vapor須至少輸入一者')
        return vapor

    vapor = check_vapor(vapor)
    return 0.622*vapor/(P-vapor)


def calc_theta_v(
        T=None, P=None, theta=None, qv=None, es=None, RH=None, Td=None, ql=0):
    '''
    計算虛位溫，可為float或陣列。
    選擇性輸入必要資訊，theta 或 (T,P)至少輸入其中之一。
    qv 或 (vapor, P) 或 (T,RH,P) 或 (Td,P)至少輸入其中之一。

    Parameters
    ----------
    T : 溫度 (K)
    P : 壓力 (Pa)
    theta : 位溫 (K)
    qv : 混和比 (kg/kg)
    es : 水氣壓 (Pa)
    RH : 相對溼度
    Td : 露點溫度 (K)
    ql : 液態水含量

    Raises
    ------
    InputError : 輸入資訊不足，詳見錯誤訊息

    Returns
    -------
    虛位溫 (K)

    '''
    def check_theta(theta):
        if theta is None:
            if type(None) not in [type(T), type(P)]:
                theta = calc_theta(T, P)
            else:
                raise InputError('theta 或 (T,P) 須至少輸入其中一者')
        return theta

    def check_qv(qv):
        if type(qv) is type(None):
            if type(None) not in [type(es), type(P)]:
                qv = calc_qv(P=P, vapor=es)
            elif type(None) not in [type(T), type(RH), type(P)]:
                qv = calc_qv(P=P, T=T, RH=RH)
            elif type(None) not in [type(Td), type(P)]:
                qv = calc_qv(P=P, Td=Td)
            else:
                raise InputError(
                    'qv 或 (vapor, P) 或 (T,RH,P) 或 (Td,P) 須至少輸入其中一者')
        return qv

    theta = check_theta(theta)
    qv = check_qv(qv)
    return theta*(1+qv*0.61-ql)


def calc_Tv(T, P=None, RH=None, vapor=None, qv=None):
    '''
    計算虛溫，可為float或陣列。
    選擇性輸入必要資訊，(P,RH) 或 (P,vapor) 或 qv至少輸入其中之一。
    Parameters
    ----------
    T : 溫度 (K)
    P : 壓力 (Pa)
    RH : 相對濕度
    vapor : 水氣壓 (Pa)
    qv : 混和比 (kg,kg)

    Returns
    -------
    float or array
        虛溫 (K)
    '''

    if type(None) not in [type(RH), type(P)]:
        return T/(1-(calc_vapor(T, RH)/P)*(1-0.622))
    elif type(None) not in [type(vapor), type(P)]:
        return T/(1-(vapor/P)*(1-0.622))
    elif type(None) not in [type(qv)]:
        return T*(1 + (qv/0.622)) / (1 + qv)
    else:
        raise InputError('(P,RH) 或 (P,vapor) 或 qv須至少輸入一者')


def calc_rho(P, Tv=None, T=None, RH=None, vapor=None, qv=None):
    '''
    計算空氣密度，可為float或陣列。
    選擇性輸入必要資訊，Tv, (T,RH) 或 (T,vapor) 或 qv至少輸入其中之一。

    Parameters
    ----------
    P : 壓力 (Pa)
    Tv : 虛溫 (K)
    T : 溫度 (K)
    RH : 相對濕度
    vapor : 水氣壓 (Pa)
    qv : 混和比 (kg,kg)

    Returns
    -------
    空氣密度 (kg/m3)
    '''
    def check_Tv(Tv):
        if type(Tv) is type(None):
            Tv = calc_Tv(T, P=P, RH=RH, vapor=vapor, qv=qv)
        return Tv

    Tv = check_Tv(Tv)
    return P/(Rd*Tv)


def calc_Tc(T, P, RH=None, vapor=None, qv=None):
    '''
    計算凝結溫度condensation temperature，可為float或陣列。
    選擇性輸入必要資訊，qv, RH 或 vapor至少輸入其中之一。
    Parameters
    ----------
    T : 溫度 (Pa)
    P : 壓力 (K)
    RH : 相對濕度
    vapor : 水氣壓 (Pa)
    qv : 混和比 (kg,kg)

    Returns
    -------
    凝結溫度 (K)
    '''
    def check_qv(qv):
        if type(qv) is type(None):
            if type(None) not in [type(RH)]:
                qv = calc_qv(P=P, T=T, RH=RH)
            elif type(None) not in [type(vapor)]:
                qv = calc_qv(P=P, vapor=vapor)
            else:
                raise InputError('qv, RH 或 vapor須至少輸入一者')
        return qv

    qv = check_qv(qv)

    Tc2 = T
    Tc = 0
    for i in range(10):
        Tc = B/(np.log(A*0.622/P/qv*(T/Tc2)**(Cp/Rd)))
        if np.max(np.abs(Tc-Tc2)) < 0.1:
            break
        Tc2 = Tc
    return Tc


def calc_theta_e(
        T=None, theta=None, P=None, RH=None, vapor=None, qv=None, Tc=None):
    '''
    計算相當位溫，可為float或陣列。
    選擇性輸入必要資訊，theta 或 (T,P)至少輸入其中之一。
    選擇性輸入必要資訊，qv 或 (T,RH,P) 或 (vapor,P)至少輸入其中之一。
    選擇性輸入必要資訊，Tc 或 P至少輸入其中之一。
    Parameters
    ----------
    T : 溫度 (Pa)
    theta : 位溫 (K)
    P : 壓力 (K)
    RH : 相對濕度
    vapor : 水氣壓 (Pa)
    qv : 混和比 (kg,kg)
    Tc : 凝結溫度 (K)

    Returns
    -------
    相當位溫 (K)
    '''
    def check_theta(theta):
        if type(theta) is type(None):
            if type(None) not in [type(T), type(P)]:
                theta = calc_theta(T, P)
            else:
                raise InputError('theta 或 (T,P)須至少輸入一者')
        return theta

    def check_qv(qv):
        if type(qv) is type(None):
            if type(None) not in [type(T), type(RH), type(P)]:
                qv = calc_qv(P, T, RH=RH)
            elif type(None) not in [type(vapor), type(P)]:
                qv = calc_qv(P, vapor=vapor)
            else:
                raise InputError('qv 或 (T,RH,P) 或 (vapor,P)須至少輸入一者')
        return qv

    def check_Tc(Tc):
        if type(Tc) is type(None):
            if type(P) is not type(None):
                Tc = calc_Tc(T, P, qv=qv)
            else:
                raise InputError('需輸入P或Tc')
        return Tc

    theta = check_theta(theta)
    qv = check_qv(qv)
    Tc = check_Tc(Tc)

    return theta*np.exp(Lv*qv/Cp/Tc)


def calc_dTdz(T=None, P=None, qvs=None):
    '''
    計算絕熱溫度遞減率，可為float或陣列。
    選擇性輸入必要資訊，不輸入T時，回傳乾絕熱遞減率，
    有輸入T時回傳指定溫度下的濕絕熱遞減率，此時P與qvs需則一輸入。
    Parameters
    ----------
    saturated : 氣塊是否飽和
    T : 溫度 (K)
    P : 壓力 (Pa)
    qvs : 飽和混和比 (kg/kg)

    Returns
    -------
    float or array
        絕熱溫度遞減率 (K/m)
    '''

    def check_qvs(qvs):
        if type(qvs) is type(None):
            if type(None) not in [type(P)]:
                qvs = calc_saturated_qv(T, P)
            else:
                raise InputError('需輸入P或qvs')
        return qvs

    if type(T) is type(None):
        return -g/Cp
    else:
        qvs = check_qvs(qvs)
        return -g*((1 + Lv*qvs/Rd/T) / (Cp + Lv**2*qvs*0.622/Rd/T**2))


def calc_RH(T=None, Th=None, vapor=None, qv=None, P=None, Td=None):
    '''
    計算相對濕度，可為float或陣列。
    選擇性輸入必要資訊，(theta, P) 或 T至少輸入其中之一。
    選擇性輸入必要資訊，vapor, Td, 或 (P, qv)至少輸入其中之一。

    Parameters
    ----------
    T : 溫度 (K)
    Th : 位溫(K)
    vapor : 水氣壓 (Pa)
    qv : 混和比 (kg,kg)
    P : 壓力 (Pa)
    Td : 露點溫度 (K)

    Returns
    -------
    相對濕度 (1)
    '''
    def check_T(T):
        if type(T) is type(None):
            if type(None) not in [type(Th), type(P)]:
                T = calc_T(Th, P)
            else:
                raise InputError('需輸入(theta, P) 或 T')
        return T

    def check_vapor(vapor):
        if type(vapor) is type(None):
            if type(None) not in [type(qv), type(P)]:
                vapor = qv_to_vapor(P, qv)
            elif type(None) is not type(Td):
                vapor = calc_vapor(Td)
            else:
                raise InputError('需輸入vapor, Td, 或 (P, qv)')
        return vapor

    T = check_T(T)
    vapor = check_vapor(vapor)
    vapor_s = calc_saturated_vapor(T)

    return vapor/vapor_s
