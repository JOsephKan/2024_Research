# This program is to define the package for computing Chien and Kim (2023)
# import ppackage
import numpy as np

# cross spectrum
def Covariance(data1, data2):
    
    data1 -= data1.mean()
    data2 -= data2.mean()
    fft1 = np.array([np.fft.fft(data1[i]) for i in range(data1.shape[0])])
    fft1 = np.array([np.fft.ifft(fft1[:, i]) for i in range(fft1.shape[1])]).T
    fft2 = np.array([np.fft.fft(data2[i]) for i in range(data2.shape[0])])
    fft2 = np.array([np.fft.ifft(fft2[:, i]) for i in range(fft2.shape[1])]).T
    cs = (fft1*fft2.conj()) / np.prod(data1.shape)
    cs_smooth = np.empty(cs.shape, dtype = complex)
    
    kernel = np.array([1, 2, 1]) / 4
    
    for i in range(cs.shape[0]):
        cs_smooth[i] = np.convolve(cs[i], kernel, mode='same')
    
    for i in range(cs.shape[1]):
        cs_smooth[:, i] = np.convolve(cs_smooth[:, i], kernel, mode='same')
    
    return cs_smooth

# growth rate
def Growth_Rate(data1, data2):
    
    var = np.array([
        Covariance(data2[i], data2[i]).real
        for i in range(data2.shape[0])
    ]).mean(axis=0)
    cov = np.array([
        Covariance(data1[i], data2[i])
        for i in range(data2.shape[0])
    ]).mean(axis=0)
    
    sigma = 2*np.real(cov) / var
    
    return sigma

# Phase
def Phase(data1, data2):

    cs = np.array([
        Covariance(data1[i], data2[i])
        for i in range(data1.shape[0])
    ]).mean(axis=0)
        
    phase = np.atan2(cs.imag, cs.real)
    
    return phase

# Coherence Square
def Coherence(data1, data2):

    var1 = np.array([
        Covariance(data1[i], data1[i]).real
        for i in range(data1.shape[0])
    ]).mean(axis=0)
    var2 = np.array([
        Covariance(data2[i], data2[i]).real
        for i in range(data2.shape[0])
    ]).mean(axis=0)
    cov  =  np.array([
        Covariance(data1[i], data2[i])
        for i in range(data1.shape[0])
    ]).mean(axis=0)
    
    Coh2 = ((cov.real)**2 + (cov.imag)**2) / (var1 * var2)
    return Coh2