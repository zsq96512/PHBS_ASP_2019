    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from . import normal
from . import bsm

'''
Asymptotic approximation for 0<beta<=1 by Hagan
'''
def bsm_vol(strike, forward, texp, sigma, alpha=0, rho=0, beta=1):
    if(texp<=0.0):
        return( 0.0 )

    powFwdStrk = (forward*strike)**((1-beta)/2)
    logFwdStrk = np.log(forward/strike)
    logFwdStrk2 = logFwdStrk**2
    rho2 = rho*rho

    pre1 = powFwdStrk*( 1 + (1-beta)**2/24 * logFwdStrk2*(1 + (1-beta)**2/80 * logFwdStrk2) )
  
    pre2alp0 = (2-3*rho2)*alpha**2/24
    pre2alp1 = alpha*rho*beta/4/powFwdStrk
    pre2alp2 = (1-beta)**2/24/powFwdStrk**2

    pre2 = 1 + texp*( pre2alp0 + sigma*(pre2alp1 + pre2alp2*sigma) )

    zz = powFwdStrk*logFwdStrk*alpha/np.fmax(sigma, 1e-32)  # need to make sure sig > 0
    if isinstance(zz, float):
        zz = np.array([zz])
    yy = np.sqrt(1 + zz*(zz-2*rho))

    xx_zz = np.zeros(zz.size)

    ind = np.where(abs(zz) < 1e-5)
    xx_zz[ind] = 1 + (rho/2)*zz[ind] + (1/2*rho2-1/6)*zz[ind]**2 + 1/8*(5*rho2-3)*rho*zz[ind]**3
    ind = np.where(zz >= 1e-5)
    xx_zz[ind] = np.log( (yy[[ind]] + (zz[ind]-rho))/(1-rho) ) / zz[ind]
    ind = np.where(zz <= -1e-5)
    xx_zz[ind] = np.log( (1+rho)/(yy[ind] - (zz[ind]-rho)) ) / zz[ind]

    bsmvol = sigma*pre2/(pre1*xx_zz) # bsm vol
    return(bsmvol[0] if bsmvol.size==1 else bsmvol)

'''
Asymptotic approximation for beta=0 by Hagan
'''
def norm_vol(strike, forward, texp, sigma, alpha=0, rho=0):
    # forward, spot, sigma may be either scalar or np.array. 
    # texp, alpha, rho, beta should be scholar values

    if(texp<=0.0):
        return( 0.0 )
    
    zeta = (forward - strike)*alpha/np.fmax(sigma, 1e-32)
    # explicitly make np.array even if args are all scalar or list
    if isinstance(zeta, float):
        zeta = np.array([zeta])
        
    yy = np.sqrt(1 + zeta*(zeta - 2*rho))
    chi_zeta = np.zeros(zeta.size)
    
    rho2 = rho*rho
    ind = np.where(abs(zeta) < 1e-5)
    chi_zeta[ind] = 1 + 0.5*rho*zeta[ind] + (0.5*rho2 - 1/6)*zeta[ind]**2 + 1/8*(5*rho2-3)*rho*zeta[ind]**3

    ind = np.where(zeta >= 1e-5)
    chi_zeta[ind] = np.log( (yy[ind] + (zeta[ind] - rho))/(1-rho) ) / zeta[ind]

    ind = np.where(zeta <= -1e-5)
    chi_zeta[ind] = np.log( (1+rho)/(yy[ind] - (zeta[ind] - rho)) ) / zeta[ind]

    nvol = sigma * (1 + (2-3*rho2)/24*alpha**2*texp) / chi_zeta
 
    return(nvol[0] if nvol.size==1 else nvol)

'''
Hagan model class for 0<beta<=1
'''
class ModelHagan:
    alpha, beta, rho = 0.0, 1.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.beta = beta
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return bsm_vol(strike, forward, texp, sigma, alpha=self.alpha, beta=self.beta, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        bsm_vol = self.bsm_vol(strike, spot, texp, sigma)
        return self.bsm_model.price(strike, spot, texp, bsm_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.bsm_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            bsm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 10)
        if(setval):
            self.sigma = sigma
        return sigma
    
    def calibrate3(self, price_or_vol3, strike3, spot, texp=1, cp_sign=1, setval=False, is_vol=True):
        '''  
        Given option prices or bsm vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''        
        if(is_vol):
            vol=price_or_vol3
        else:
            vol=np.zeros(3)
            vol[0]=self.bsm_model.impvol(price_or_vol3[0], strike3[0], 100, 1, cp_sign=cp_sign)
            vol[1]=self.bsm_model.impvol(price_or_vol3[1], strike3[1], 100, 1, cp_sign=cp_sign)
            vol[2]=self.bsm_model.impvol(price_or_vol3[2], strike3[2], 100, 1, cp_sign=cp_sign)
        def fun(x):
            return bsm_vol(strike3, spot, texp, x[0], x[1], x[2]) - vol
        
        para=sopt.root(fun,[0.1,0.2,0.15])
        
        return para.x   # sigma, alpha, rho


'''
Hagan model class for beta=0
'''
class ModelNormalHagan:
    alpha, beta, rho = 0.0, 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.beta = 0.0 # not used but put it here
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return norm_vol(strike, forward, texp, sigma, alpha=self.alpha, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        n_vol = self.norm_vol(strike, spot, texp, sigma)
        return self.normal_model.price(strike, spot, texp, n_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            norm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 50)
        if(setval):
            self.sigma = sigma
        return sigma

    def calibrate3(self, price_or_vol3, strike3, spot, texp=1, cp_sign=1, setval=False, is_vol=True):
        '''  
        Given option prices or normal vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''
        if(is_vol):
            vol=price_or_vol3
        else:
            vol=np.zeros(3)
            vol[0]=self.normal_model.impvol(price_or_vol3[0], strike3[0], 100, 1, cp_sign=cp_sign)
            vol[1]=self.normal_model.impvol(price_or_vol3[1], strike3[1], 100, 1, cp_sign=cp_sign)
            vol[2]=self.normal_model.impvol(price_or_vol3[2], strike3[2], 100, 1, cp_sign=cp_sign)
        def fun(x):
            return norm_vol(strike3, spot, texp, x[0], x[1], x[2]) - vol
        
        para=sopt.root(fun,[19,0.1,-0.1])
        
        return para.x   # sigma, alpha, rho

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    time_step=100
    NumSim=1000
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        price=self.price(strike=strike,spot=spot,texp=self.texp,sigma=self.sigma,cp_sign=1)
        vol=self.bsm_model.impvol(price_in=price, strike=strike, spot=spot, texp=self.texp, cp_sign=1)
        return vol
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        NumSim=self.NumSim
        time_step=self.time_step
        rho=self.rho
        sigma = self.sigma
        alpha = self.alpha        
        norm_z=np.random.normal(size=(NumSim,time_step))
        norm_w=rho*norm_z+np.sqrt(1-rho**2)*np.random.normal(size=(NumSim,time_step))
        vec_sigma=np.zeros(shape=(NumSim,time_step))
        vec_price=np.zeros(shape=(NumSim,time_step))
        for i in range(0,time_step):
            if i==0:
                vec_sigma[:,i]=sigma * np.exp(alpha*np.sqrt(1/time_step)*norm_z[:,i]-0.5*alpha**2*1/time_step)
                vec_price[:,i]=np.log(spot)+vec_sigma[:,i]*norm_w[:,i]*np.sqrt(1/time_step)-0.5*vec_sigma[:,i]**2*1/time_step
            else:
                vec_sigma[:,i]=vec_sigma[:,i-1]*np.exp(alpha*np.sqrt(1/time_step)*norm_z[:,i]-0.5*alpha**2*1/time_step)
                vec_price[:,i]=vec_price[:,i-1]+vec_sigma[:,i]*norm_w[:,i]*np.sqrt(1/time_step)-0.5*vec_sigma[:,i]**2*1/time_step
        price=np.zeros(shape=np.size(strike))
        var=np.zeros(shape=np.size(strike))
        for i in range(0,np.size(strike)):
            
            price[i]=np.mean(np.fmax(0,np.exp(vec_price[:,-1])-strike[i]))
            var[i]=np.std(np.fmax(0,np.exp(vec_price[:,-1])-strike[i]))
        return price,var     

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    time_step=100
    NumSim=1000
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        price=self.price(strike=strike,spot=spot,texp=self.texp,sigma=self.sigma,cp_sign=1)
        vol=self.normal_model.impvol(price_in=price, strike=strike, spot=spot, texp=self.texp, cp_sign=1)
        return vol

        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        #correlated Normal r.v.
        #1000 rows, 100columns
        NumSim=self.NumSim
        time_step=self.time_step
        rho=self.rho
        sigma = self.sigma
        alpha = self.alpha    
        norm_z=np.random.normal(size=(NumSim,time_step))
        norm_w=rho*norm_z+np.sqrt(1-rho**2)*np.random.normal(size=(NumSim,time_step))
        vec_sigma=np.zeros(shape=(NumSim,time_step))
        vec_price=np.zeros(shape=(NumSim,time_step))
        for i in range(0,time_step):
            if i==0:
                vec_sigma[:,i]=sigma * np.exp(alpha*np.sqrt(1/time_step)*norm_z[:,i]-0.5*alpha**2*1/time_step)
                vec_price[:,i]=spot +vec_sigma[:,i]*norm_w[:,i]*np.sqrt(1/time_step)
            else:
                vec_sigma[:,i]=vec_sigma[:,i-1]*np.exp(alpha*np.sqrt(1/time_step)*norm_z[:,i]-0.5*alpha**2*1/time_step)
                vec_price[:,i]=vec_price[:,i-1]+vec_sigma[:,i]*norm_w[:,i]*np.sqrt(1/time_step)
        price=np.zeros(shape=np.size(strike))
        var=np.zeros(shape=np.size(strike))
        for i in range(0,np.size(strike)):
            
            price[i]=np.mean(np.fmax(vec_price[:,-1]-strike[i],0))
            var[i]=np.std(np.fmax(vec_price[:,-1]-strike[i],0))
        return price,var

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    time_step=100
    NumSim=1000    
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        price=self.price(strike=strike,spot=spot,texp=self.texp,sigma=self.sigma,cp_sign=1)
        vol=self.bsm_model.impvol(price=price, strike=strike, spot=spot, texp=self.texp, cp_sign=1)
        return vol

    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        NumSim=self.NumSim
        time_step=self.time_step
        rho=self.rho
        sigma = self.sigma
        alpha = self.alpha  
        texp=self.texp
        norm_z=np.random.normal(size=(NumSim,time_step))
        vec_sigma=np.zeros(shape=(NumSim,time_step))
        for i in range(0,time_step): 
            if i==0:
                vec_sigma[:,i]=sigma * np.exp(alpha*np.sqrt(1/time_step)*norm_z[:,i]-0.5*alpha**2*1/time_step)             
            else:
                vec_sigma[:,i]=vec_sigma[:,i-1]*np.exp(alpha*np.sqrt(1/time_step)*norm_z[:,i]-0.5*alpha**2*1/time_step)
        I_T=(np.sum(2*vec_sigma**2,axis=1)-vec_sigma[:,0]**2-vec_sigma[:,-1]**2)*1/time_step/2
        spot=spot*np.exp(rho/alpha*(vec_sigma[:,-1]-sigma)-rho**2/2*I_T)
        sigma=np.sqrt((1-rho**2)*I_T/texp)
        price=np.zeros(shape=np.size(strike))  
        var=np.zeros(shape=np.size(strike))
        import option_models as opt
        for i in range(0,np.size(strike)):
            price[i]=np.mean(bsm.price(strike=strike[i],spot=spot,texp=texp,vol=sigma))
            var[i]=np.std(bsm.price(strike=strike[i],spot=spot,texp=texp,vol=sigma))
        return price,var

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    time_step=100
    NumSim=1000
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        price=self.price(strike=strike,spot=spot,texp=self.texp,sigma=self.sigma,cp_sign=1)
        vol=self.bsm_model.impvol(price_in=price, strike=strike, spot=spot, texp=self.texp, cp_sign=1)
        return vol
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        np.random.seed(12345)
        NumSim=1000
        time_step=100
        rho=-0.25
        sigma = 20
        alpha = 0.5 
        texp=1
        spot=100
        norm_z=np.random.normal(size=(NumSim,time_step))
        vec_sigma=np.zeros(shape=(NumSim,time_step))
        for i in range(0,time_step): 
            if i==0:
                vec_sigma[:,i]=sigma * np.exp(alpha*np.sqrt(1/time_step)*norm_z[:,i]-0.5*alpha**2*1/time_step)             
            else:
                vec_sigma[:,i]=vec_sigma[:,i-1]*np.exp(alpha*np.sqrt(1/time_step)*norm_z[:,i]-0.5*alpha**2*1/time_step)
        spot=spot+rho/alpha*(vec_sigma[:,-1]-sigma)
        I_T=(np.sum(2*vec_sigma**2,axis=1)-vec_sigma[:,0]**2-vec_sigma[:,-1]**2)*1/time_step/2
        sigma=np.sqrt((1-rho**2)*I_T/texp)
        price=np.zeros(shape=np.size(strike))
        var=np.zeros(shape=np.size(strike))
        for i in range(0,np.size(strike)):
            price[i]=np.mean(normal.price(strike=strike[i],spot=spot,texp=texp,vol=sigma))
            var[i]=np.std(normal.price(strike=strike[i],spot=spot,texp=texp,vol=sigma))
        return price,var
