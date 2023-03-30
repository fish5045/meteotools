from ..calc import calc_Tv, calc_rho, calc_theta, \
    calc_T, calc_saturated_vapor, calc_vapor, calc_saturated_qv, qv_to_vapor, \
    calc_theta_es, calc_Td, calc_qv, calc_theta_v, calc_Tv, calc_rho, \
    calc_Tc, calc_theta_e, calc_dTdz, calc_RH


class calc_thermaldynamic:
    def calc_theta(self):
        if 'T' in dir(self) and 'p' in dir(self):
            self.Th = calc_theta(self.T, self.p)
        else:
            raise AttributeError('須先設定T, p')

    def calc_T(self):
        if 'Th' in dir(self) and 'p' in dir(self):
            self.T = calc_T(self.Th, self.p)
        else:
            raise AttributeError('須先設定Th, p')

    def calc_saturated_vapor(self):
        if 'T' not in dir(self):
            self.calc_T()
        self.vapor_s = calc_saturated_vapor(self.T)

    def calc_saturated_qv(self):
        if 'T' not in dir(self):
            self.calc_T()
        if 'p' in dir(self):
            self.qvs = calc_saturated_qv(self.T, self.p)
        else:
            raise AttributeError('須先設定p')

    def calc_vapor(self):
        if 'Td' in dir(self):
            self.vapor = calc_vapor(self.Td)
        elif 'qv' in dir(self) and 'p' in dir(self):
            self.vapor = qv_to_vapor(self.p, self.qv)
        elif 'T' in dir(self) and 'RH' in dir(self):
            self.vapor = calc_vapor(self.T, self.RH)
        elif all(var in dir(self) for var in ['Th', 'RH', 'p']):
            self.calc_T()
            self.vapor = calc_vapor(self.T, self.RH)
        else:
            raise AttributeError(
                '須先設定(Th, p, RH), (p, qv), (Td), 或 (T, RH)其一。')

    def calc_Td(self):
        if 'vapor' not in dir(self):
            self.calc_vapor()
        self.Td = calc_Td(es=self.vapor)

    def calc_qv(self):
        if 'vapor' not in dir(self):
            self.calc_vapor()
        if 'p' in dir(self):
            self.qv = calc_qv(self.p, vapor=self.vapor)
        else:
            raise AttributeError('須先設定p')

    def calc_theta_es(self):
        if 'T' not in dir(self):
            self.calc_T()
        if 'p' in dir(self):
            self.Th_es = calc_theta_es(self.T, self.p)
        else:
            raise AttributeError('須先設定p')

    def calc_theta_v(self):
        if 'Th' not in dir(self):
            self.calc_theta()
        if 'qv' not in dir(self):
            self.calc_qv()
        if 'qc' in dir(self) and 'qr' in dir(self):
            self.Th_v = calc_theta_v(
                theta=self.Th, qv=self.qv, ql=self.qc+self.qr)
        elif 'qc' in dir(self):
            self.Th_v = calc_theta_v(theta=self.Th, qv=self.qv, ql=self.qc)
        elif 'qr' in dir(self):
            self.Th_v = calc_theta_v(theta=self.Th, qv=self.qv, ql=self.qr)
        else:
            self.Th_v = calc_theta_v(theta=self.Th, qv=self.qv)

    def calc_Tv(self):
        if 'T' not in dir(self):
            self.calc_T()
        if 'qv' not in dir(self):
            self.calc_qv()
        self.Tv = calc_Tv(self.T, qv=self.qv)

    def calc_rho(self):
        if 'self.Tv' not in dir(self):
            self.calc_Tv()
        if 'p' in dir(self):
            self.rho = calc_rho(self.p, Tv=self.Tv)
        else:
            raise AttributeError('須先設定p')

    def calc_Tc(self):
        if 'T' not in dir(self):
            self.calc_T()
        if 'qv' not in dir(self):
            self.calc_qv()
        if 'p' in dir(self):
            self.Tc = calc_Tc(self.T, self.p, qv=self.qv)
        else:
            raise AttributeError('須先設定p')

    def calc_theta_e(self):
        if 'Th' not in dir(self):
            self.calc_theta()
        if 'qv' not in dir(self):
            self.calc_qv()
        if 'Tc' not in dir(self):
            self.calc_Tc()
        self.Th_e = calc_theta_e(theta=self.Th, qv=self.qv, Tc=self.Tc)

    def calc_moist_dTdz(self):
        if 'T' not in dir(self):
            self.calc_T()
        if 'p' in dir(self) and 'qvs' in dir(self):
            self.moist_dTdz = calc_dTdz(qvs=self.qvs)
        else:
            self.moist_dTdz = calc_dTdz(T=self.T, P=self.p)

    def calc_dry_dTdz(self):
        self.dry_dTdz = calc_dTdz()

    def calc_RH(self):
        if 'T' not in dir(self):
            self.calc_T()
        if 'vapor' not in dir(self):
            self.calc_vapor()
        self.RH = calc_RH(T=self.T, vapor=self.vapor)
