import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp, LSODA
from scipy.ndimage import gaussian_filter1d as gf
from scipy.interpolate import interp1d
import pickle
from scipy.optimize import curve_fit

class TES:
    
    def __init__( self, Rn=176e-3, Tc=51e-3, Tw=0.5e-3, Tb=13e-3, G=4.6e-11, L=130e-9, Rsh=1.8e-3, Rp=2.3e-3, C=4e-14, r_param='tanh', tau_ph=20e-6, coll_eff=1 ):
        self.Rn = Rn
        self.Tc = Tc
        self.Tw = Tw
        self.Tb = Tb
        self.G = G
        self.L = L
        self.Rp = Rp
        self.Rsh = Rsh
        self.Rl = (Rp+Rsh)
        self.C = C
        self.tau_ph = tau_ph
        self.coll_eff = coll_eff
        self.r_param=r_param
        self.kappa = G/5/Tc**4
        self.Vb = np.nan
        self.T0 = np.nan
        self.I0 = np.nan
        self.R0 = np.nan
        self.alpha = np.nan
        self.lgain = np.nan
        self.tau0 = np.nan
        self.tau_rise = np.nan
        self.tau_fall = np.nan
        self.tau_el = np.nan
        self.tau_I = np.nan
        self.P0 = np.nan
        
    def set_Vb( self, Vb ):
        self.Vb = Vb
        
    def set_derived_params(self):
    
        self.alpha = (2*self.T0/self.Tw)*( 1 - self.R0/self.Rn )
        self.P0 = self.I0**2*self.R0
        self.lgain = self.P0*self.alpha/(self.G*self.T0)
        self.tau0 = self.C/self.G
        self.tau_el = self.L/( self.Rl + self.R0 )
        self.tau_I = self.tau0/( 1-self.lgain )
        
        lambda_mean = 0.5*( 1./self.tau_el+1./self.tau_I )
        lambda_discrim = 0.5*np.sqrt( (1./self.tau_el-1./self.tau_I)**2-8*self.R0*self.lgain/self.L/self.tau0 )
        
        self.tau_rise = 1./( lambda_mean + lambda_discrim )
        self.tau_fall = 1./( lambda_mean - lambda_discrim )
        
    def get_kappa( self ):
        self.kappa = self.G/5/self.Tc**4
        
    def R_TES( self, T ):
        # return the resistence of the TES given a temperature
        if self.r_param not in ['tanh', 'compact']:
            print( 'r_param must be either \'tanh\' or \'compact\' ' )
            return np.nan
        elif self.r_param=='tanh':
            return 0.5*self.Rn*(1+np.tanh((T-self.Tc)/self.Tw))
        else:
            linear_on = np.heaviside(T-(self.Tc-1*self.Tw), 0)*np.heaviside((self.Tc+1*self.Tw)-T, 0)
            normal_on = np.heaviside(T-(self.Tc+1*self.Tw), 1)
            linear_arg = (self.Rn/(2*self.Tw))*(T-self.Tc)+self.Rn/2
            return linear_on*linear_arg + self.Rn*normal_on
    
    def eq_condition( self, T ):
        # roots of this function give us T0
        return self.Vb**2*self.R_TES(T)/(self.R_TES(T)+self.Rl)**2 - self.kappa*(T**5-self.Tb**5)
    
    def find_eq( self ):
        
        #s = root_scalar(self.eq_condition, x0=self.Tc-2*self.Tw, x1=self.Tc+2*self.Tw, xtol=1e-10)
        # first find where R(T) = R_L
        Tgrid = np.linspace(self.Tc-5*self.Tw, self.Tc+5*self.Tw, 100 )
        Rgrid = self.R_TES(Tgrid)
        R_interp = interp1d( Tgrid, Rgrid-self.Rl, fill_value='extrapolate' )
        s_Rl = root_scalar( R_interp, bracket=[self.Tc-10*self.Tw, self.Tc+10*self.Tw], method='brentq' )
        
        try:
            s = root_scalar(self.eq_condition, bracket=[s_Rl.root, self.Tc+100*self.Tw], method='brentq', xtol=1e-14)
            self.T0=s.root
        except:
            self.T0=self.Tb

        self.R0 = self.R_TES(self.T0)
        self.I0 = self.Vb/(self.R0+self.Rl)
        self.Tsat = self.Tc+self.Tw-self.T0
        
    def dIdt(self, T, I ):
        return (1./self.L)*(self.Vb-I*(self.R_TES(T)+self.Rl))
    
    def dTdt(self, T, I ):
        return (1./self.C)*(I**2*self.R_TES(T)-self.kappa*(T**5-self.Tb**5))
    
    def my_system_homo(self, t, y ):
        T, I = y
        return np.array([self.dTdt(T,I), self.dIdt(T,I)])

    def my_system_driven(self, t, y, tau_phonon, delta_T ):
        T, I = y

        temp_input = (delta_T/tau_phonon)*np.exp(-t/tau_phonon)
        
        return np.array([self.dTdt(T,I)+temp_input, self.dIdt(T,I)])
    
    def evolve_system_inst(self, delta_E ):
        # input energy in Joules!
        
        tstop = 10e-3
        tgrid = np.linspace(0, tstop, 100000)
        
        delta_T = (delta_E)/self.C
        #print(delta_T, self.T0+delta_T)
        sol = solve_ivp( self.my_system_homo, (0, tstop), np.array([self.T0+delta_T, self.I0]), t_eval=tgrid, method='LSODA', vectorized=True, max_step=1e-5 )
        sol_T = sol.y[0,:]
        sol_I = sol.y[1,:]
        
        ret_dict = {'t':tgrid, 'dI':sol_I-self.I0, 'dT':sol_T-self.T0}
        return ret_dict
    
    def evolve_system_delay(self, delta_E ):
        # input energy in Joules!
        
        tstop = 10e-3
        tgrid = np.linspace(0, tstop, 1000000)
        
        delta_T = (delta_E)/self.C
        #tau_ph = 20e-6
        
        sol = solve_ivp( self.my_system_driven, (0, tstop), np.array([self.T0, self.I0]), args=(self.tau_ph, delta_T), t_eval=tgrid, method='LSODA', vectorized=True, max_step=1e-5 )
        sol_T = sol.y[0,:]
        sol_I = sol.y[1,:]
        
        ret_dict = {'t':tgrid, 'dI':sol_I-self.I0, 'dT':sol_T-self.T0}
        return ret_dict
    
    def integrate_energy( self, response ):
        
        T_trace = response['dT']+self.T0
        I_trace = response['dI']+self.I0
        
        J_power_trace = np.power( I_trace, 2 )*self.R_TES( T_trace ) - self.I0**2*self.R0
        B_power_trace = self.kappa*(np.power( T_trace, 5 ) - self.T0**5 )
        
        delta_EJ = -np.trapz( J_power_trace, response['t'] )
        delta_EB = np.trapz( B_power_trace, response['t'] )
        
        return { 'dEJ':delta_EJ, 'dEB':delta_EB }
    