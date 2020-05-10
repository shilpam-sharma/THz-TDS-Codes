# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:00:58 2019

This script performs the de-noising of THz-TDS data.
The script first removes the DC slope from air, film+sample & sample
data by high pass filtering and then removes the background data collected with
blank sample holder. The script next generates a window centered around the max
peak of signal & then performs the FFT of raw data and the windowed data.
It finally saves the FFT of the data in a new folder called 'FFT' in PWD.
The script saves magnitude & phase of the raw and windowed data in separate
files. The script tries to optimize the n, k of the sample and film using
Kida formalism PRB 62, R11966 and saves the data for the electrical params in
separate folders for the raw and windowed data in the the PWD

@Author: Dr. Shilpam Sharma
Last edit: 8/5/2020, 09.23 pm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.optimize as opt
from scipy import signal
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.constants import c, pi, epsilon_0
import os
import pathlib
import pickle


'''**************** File paths for data and backgnd here**************'''
path = r'C:/Users/A/Desktop/Ashish data'
currentDirectory = pathlib.Path(path)

'''########### de-noising constants/prams defined here #########'''

f_min = 0.20e12             # minimum relevent frequency for data truncation
f_max = 1.2e12              # maximum relevent frequency for data truncation
lowcut = 0.05e12            # low cutoff of band pass butterworth filter
highcut = 1.4e12            # high cutoff of band pass butterworth filter
order = 5                   # order of the High pass butterworth filter
win_pts = 20              # no. of points in the window around signal
beta_air = 10               # film and susbtarte shape factor for Kaiser window
beta_f_s = 10                # film and susbtarte shape factor for Kaiser window
data_file_header = True     # data file contains header or not
#bck_file_header = True     # background file contains header or not
filtering = True
plotting = True            # plotting of different quantities
'''######### optimization constants defined here ###############'''
sampl = 'Mn0.8Zn0.2Fe2O4'
d_sample = 0.87e-3           # susbtrate thickness in mts
methods = 'NM'              # NM for Nelder-Mead or BFGS for L-BFGS-B algol

data_fft_raw = pd.DataFrame()
'''dumping parameters to a binary file for later use'''
denoise_param = []
denoise_param.append(['f_min', 'f_max', 'lowcut', 'highcut', 'order', 'win_pts', 'beta_air', 'beta_f_s','d_sample'])
denoise_param.append([f_min, f_max, lowcut, highcut, order, win_pts, beta_air, beta_f_s, d_sample])
denoise_para_directory = os.path.join(currentDirectory, 'DenoiseParameters')
if not os.path.exists(denoise_para_directory):
    os.makedirs(denoise_para_directory)
pickledParamFile = os.path.join(currentDirectory, 'DenoiseParameters', 'denoiseParameters.dnp')
ParamTxtFile = open(os.path.join(currentDirectory, 'DenoiseParameters', 'denoiseParameters.txt'), "w")
pickle.dump(denoise_param, open(pickledParamFile, "wb"))
for i in range(len(denoise_param[0])):
    print(denoise_param[0][i], ': ', denoise_param[1][i])
    ParamTxtFile.write(denoise_param[0][i])
    ParamTxtFile.write(' : \t')
    ParamTxtFile.write(str(denoise_param[1][i]))
    ParamTxtFile.write('\n')
ParamTxtFile.close()

'''######### Definening the High pass butterworth filter for removing the slopes from E vs t data #########'''
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    data_hpfilt = signal.sosfiltfilt(sos, data, padtype='even', padlen=None)
    return data_hpfilt
'''*********************** High pass filter definition over ***********************************************'''

def fitPhaseStLine(freq, m, c):			# the fit function for phase correction
    return m*freq+c


'''################# Data reading starts here ###################'''

currentPattern = "*.dat"
for currentFile in currentDirectory.glob(currentPattern):
    work_dir, cur_file = os.path.split(currentFile)
    idf = cur_file.split("_")

    for i in idf:
        if i.find('K')!= -1:    # find temperature info from file name
            temperature = i
            t = temperature.split('K')[0]
            if t.find('p')!= -1:
                t = float(t.replace('p', '.'))
            else:
                t = float(t)

    file_air = cur_file

    path_air = os.path.join(currentDirectory, file_air)

    print(temperature, '\n')
    # read data files
    if data_file_header == True:
        ''' uncomment here if file contains column names as header'''
        data_sample = pd.read_csv(path_air, delimiter = "\t", usecols = ['Interval', 'Data Level'])
        data_air = pd.read_csv(path_air, delimiter = "\t", usecols = ['Interval', 'Ref Level'])
    else:
        '''use below lines to read the files without header info'''
        names = ['Interval', 'Data Level']
        data_sample = pd.read_csv(path_air, delimiter = "\t", header = None, usecols = [0, 2], names = ['Interval', 'Data Level'])
        data_air = pd.read_csv(path_air, delimiter = "\t", header = None, usecols = [0, 1], names = ['Interval', 'Ref Level'])


    '''################ High pass filtering of the air data starts here ##########################'''
    # E vs t slope correction for air data
    x_air = data_air['Interval']
    y_air = data_air['Ref Level']
    fs = 1/x_air[1]
    y_air_hpf = butter_bandpass_filter(y_air, lowcut, highcut, fs, order)

    ''' High pass filtering of the sample data starts here'''

    x_sample = data_sample['Interval']
    y_sample = data_sample['Data Level']
    fs = 1/x_sample[1]
    y_sample_hpf = butter_bandpass_filter(y_sample, lowcut, highcut, fs, order)

    if filtering == True:
        y_air_corr = pd.Series(y_air_hpf)                       #convert back to Pandas series)
        y_sample_corr = pd.Series(y_sample_hpf)
    else:
        y_air_corr = pd.Series(y_air)                           #convert back to Pandas series)
        y_sample_corr = pd.Series(y_sample)

    '''Air FFT to begin'''
    sp_air_raw = fftshift(fft(y_air_corr))
    freq_air_raw = fftshift(fftfreq(x_air.size, x_air[1]))
    mag_air_raw = np.abs(sp_air_raw)
    phase_air_raw = np.unwrap(np.angle(sp_air_raw))

    '''Sample FFT begin here'''
    sp_sample_raw = fftshift(fft(y_sample_corr))
    freq_sample_raw = fftshift(fftfreq(x_sample.size, x_sample[1]))
    mag_sample_raw = np.abs(sp_sample_raw)
    phase_sample_raw = np.unwrap(np.angle(sp_sample_raw))

    '''Truncate the results to the relevent frequency range of f_min THz to f_max THz'''
    '''for air'''
    index_f_max = np.where(freq_air_raw<=f_max)[0][-1]
    index_f_min = np.where(freq_air_raw>=f_min)[0][0]
    freq_air_thz = freq_air_raw[index_f_min:index_f_max]
    sp_air_raw_thz = sp_air_raw[index_f_min:index_f_max]
    mag_air_raw_thz = mag_air_raw[index_f_min:index_f_max]
    phase_air_raw_thz = phase_air_raw[index_f_min:index_f_max]
    ''' for sample'''
    index_f_max = np.where(freq_sample_raw<=f_max)[0][-1]
    index_f_min = np.where(freq_sample_raw>=f_min)[0][0]
    freq_sample_thz = freq_sample_raw[index_f_min:index_f_max]
    sp_sample_raw_thz = sp_sample_raw[index_f_min:index_f_max]
    mag_sample_raw_thz = mag_sample_raw[index_f_min:index_f_max]
    phase_sample_raw_thz = phase_sample_raw[index_f_min:index_f_max]

    '''Transmission of Sample with respect to air '''
    Trans_sample_air_raw = mag_sample_raw_thz/mag_air_raw_thz
    phase_diff_sample_air_raw = phase_sample_raw_thz-phase_air_raw_thz

    optimizedParameters, pcov = opt.curve_fit(fitPhaseStLine, freq_sample_thz, phase_diff_sample_air_raw, method='trf', bounds=((-np.inf, -np.inf), (np.inf, np.inf)))
    phase_diff_sample_air_raw_corrected = phase_diff_sample_air_raw-optimizedParameters[1]

    '''The initial guess of n and k for whole frequency range is calculated here'''
    nsa = np.abs(1-((c/(2*pi*freq_sample_thz*d_sample))*phase_diff_sample_air_raw_corrected))
    ksa = np.log((4*nsa)/((nsa+1)*Trans_sample_air_raw))*(c/(2*pi*freq_sample_thz*d_sample))

    for r in range(len(freq_air_thz)):
        data_fft_raw.loc[r, 'Freq (cm-1)'] = freq_air_thz[r]/2.998e10
        data_fft_raw.loc[r, 'Freq (Hz)'] = freq_air_thz[r]
        data_fft_raw.loc[r, 'Magnitude (air)'] = mag_air_raw_thz[r]
        data_fft_raw.loc[r, 'Phase (air)'] = phase_air_raw_thz[r]
        data_fft_raw.loc[r, 'Freq (sample)'] = freq_sample_thz[r]
        data_fft_raw.loc[r, 'Magnitude (sample)'] = mag_sample_raw_thz[r]
        data_fft_raw.loc[r, 'Phase (sample)'] = phase_sample_raw_thz[r]
        data_fft_raw.loc[r, 'Tr (sample_air)'] = Trans_sample_air_raw[r]
        data_fft_raw.loc[r, 'dPh (sample_air)'] = phase_diff_sample_air_raw[r]

    temp_out = np.full(len(freq_air_thz), t)
    file_out_raw = "{filename}_{samp}_{ids}.{ext}".format(filename=temperature, samp = sampl, ids='FFT_raw', ext=idf[-1].split('.')[-1])
    file_out_win = "{filename}_{samp}_{ids}.{ext}".format(filename=temperature, samp = sampl, ids='FFT_win', ext=idf[-1].split('.')[-1])
    file_out_p_raw = "{filename}_{samp}_{ids}.{ext}".format(filename=temperature, samp = sampl, ids='EleParams_raw', ext=idf[-1].split('.')[-1])
    file_out_p_win = "{filename}_{samp}_{ids}.{ext}".format(filename=temperature, samp = sampl, ids='EleParams_win', ext=idf[-1].split('.')[-1])
    new_directory_raw = os.path.join(currentDirectory, 'FFT Data_raw')
    new_directory_p_raw = os.path.join(currentDirectory, 'Ele_params_raw')

    if not os.path.exists(new_directory_raw):
        os.makedirs(new_directory_raw)
    if not os.path.exists(new_directory_p_raw):
        os.makedirs(new_directory_p_raw)

    path_out_raw = os.path.join(currentDirectory, 'FFT Data_raw', file_out_raw)
    data_fft_raw.to_csv(path_out_raw, index=False, sep='\t', header=data_fft_raw.columns.tolist())


    '''*********************************************************'''
    '''########## Optimisation of n, k starts here #############'''
    '''*********************************************************'''
    data_fft_name = [data_fft_raw] # for running the optimization on both raw and windowed data
    dir_names = ['Ele_params_raw']    #give different directories for raw and windowed data
    file_names = [file_out_p_raw]
    for name in range(len(data_fft_name)):

        '''some makeup for the optimized data containers'''
        n_sample = np.empty(len(data_fft_name[name]['Freq (Hz)']))
        k_sample = np.empty(len(data_fft_name[name]['Freq (Hz)']))
        data_Ele_param = pd.DataFrame()

        '''definition of the transfer function of air to sample'''
        def transfer_sample(f, nk_sample):
            N2 = nk_sample[0]-1j*nk_sample[1]
            t3 = (2*N2)/(1+N2)
            t3_p = 2/(N2+1)
            k0 = (2*pi*f)/c
            k2 = k0*N2
            transmission = (t3*t3_p)*np.exp(-1j*(k2-k0)*d_sample)
            return transmission

        """ cost function of air to sample for minimization"""
        def error_fun_sample(nk_sample, f, tr_exp_sample):
            return np.linalg.norm(tr_exp_sample-transfer_sample(f, nk_sample))

        '''fitting of the experimental transfer function of the sample to theoretical model'''
        def match_sample(f,tr_exp_sample, nk_sample0):
            s_opt = minimize(error_fun_sample, nk_sample0, args=(f, tr_exp_sample), method='Nelder-Mead', tol = 1e-8, options={'return_all':False, 'adaptive': True, 'disp': True, 'maxiter':1000000, 'maxfev':100000, 'xatol': 0.00000001, 'fatol': 0.00000001})
            # bound = ((0, nk_sample0[0]), (0.0, nk_sample0[1]))
            # s_opt = minimize(error_fun_sample, nk_sample0, args=(f, tr_exp_sample), method='SLSQP', tol = 1e-8, bounds= bound)
            return s_opt
        '''calculation of the experimental transmission of air-sample'''

        tr_exp_sample = data_fft_name[name]['Tr (sample_air)']*np.exp(1j*(data_fft_name[name]['dPh (sample_air)']))

        #nk_sample0 = [nsa[0], ksa[0]]
        #'''do minimization of n & k of sample for each frequency'''
        for i in range(len(data_fft_name[name]['Freq (Hz)'])):
            f = data_fft_name[name]['Freq (Hz)'][i]
            nk_sample0 = [nsa[i], ksa[i]]
            nk_opt_sample = match_sample(f, tr_exp_sample[i], nk_sample0) # sample minimization
            n_sample[i], k_sample[i] = nk_opt_sample.x
            nk_sample = nk_opt_sample.x
            #nk_sample0 = nk_sample.tolist()



        '''conductivity and dielctric constant calculations'''
        '''samplestrate'''
        eps1_sample = (n_sample**2)-(k_sample**2)
        eps1_sample_est = (nsa**2)-(ksa**2)
        eps2_sample = 2.0*(n_sample*k_sample)
        eps2_sample_est = 2.0*(nsa*ksa)

        sigma1_sample = 2.0*epsilon_0*(2.0*pi*data_fft_name[name]['Freq (Hz)'])*(n_sample*k_sample)
        sigma1_sample_est = 2.0*epsilon_0*(2.0*pi*data_fft_name[name]['Freq (Hz)'])*(nsa*ksa)
        sigma2_sample = epsilon_0*(2.0*pi*data_fft_name[name]['Freq (Hz)'])*(1.0-(n_sample**2)+(k_sample**2))
        '''film'''


        data_Ele_param.insert(0,'Freq (Hz)', data_fft_name[name]['Freq (Hz)'])
        data_Ele_param.insert(1, 'Temperature (K)', temp_out)
        data_Ele_param.insert(2,'n sample', np.abs(n_sample))
        data_Ele_param.insert(3,'k sample', np.abs(k_sample))
        data_Ele_param.insert(4,'Sigma1_sample (S/cm)', sigma1_sample/100.0)
        data_Ele_param.insert(5,'Sigma2_sample (S/cm)', sigma2_sample/100.0)
        data_Ele_param.insert(6,'Eps1_sample', eps1_sample)
        data_Ele_param.insert(7,'Eps2_sample', eps2_sample)

        path_out = os.path.join(currentDirectory, dir_names[name], file_names[name])
        data_Ele_param.to_csv(path_out, index=False, sep='\t', header=data_Ele_param.columns.tolist())

        '''**********### Data crunching ends here ###***************'''

    '''let's plot somethings for showoff purposes'''
    if plotting == True:
        plt.figure(num=1)
        plt.subplot(121)
        plt.title('E vs t plot')
        plt.xlabel('t (sec.)')
        plt.ylabel('E (a. u.)')
        plt.plot(x_air, y_air_corr)
        plt.plot(x_sample, y_sample_corr)
        plt.gca().legend(('air', 'sample'))
        plt.subplot(222)
        plt.title('FFT of Air signal')
        plt.xlabel('f (Hz)')
        plt.ylabel('magnitude (a. u.)')
        plt.semilogy(freq_air_thz, mag_air_raw_thz, 'o')
        plt.gca().legend(('raw data', 'window'))
        plt.tight_layout()
        plt.subplot(224)
        plt.title('FFT of Sample signal')
        plt.xlabel('f (Hz)')
        plt.ylabel('magnitude (a. u.)')
        plt.semilogy(freq_sample_thz, mag_sample_raw_thz, 'o')
        plt.gca().legend(('raw data', 'window'))
        plt.tight_layout()
        plt.pause(0.2)

        plt.figure(num=2)
        plt.subplot(121)
        plt.semilogy(freq_air_thz, mag_air_raw_thz, freq_sample_thz, mag_sample_raw_thz)
        plt.xlabel('f (Hz)')
        plt.ylabel('magnitude (a. u.)')
        plt.gca().legend(('air', 'Sample'))
        plt.subplot(122)
        plt.plot(freq_sample_thz, Trans_sample_air_raw)
        plt.xlabel('f (Hz)')
        plt.ylabel('transmission (a.u.)')
        plt.gca().legend(('Sample-air'))
        plt.tight_layout()
        plt.pause(0.2)

        plt.figure(num=3)
        plt.subplot(121)
        plt.title('phase of raw data')
        plt.plot(freq_air_thz, phase_air_raw_thz, freq_sample_thz, phase_sample_raw_thz)
        plt.xlabel('f (Hz)')
        plt.ylabel('phase (radians)')
        plt.gca().legend(('air', 'Sample'))


        plt.subplot(222)
        plt.plot(freq_sample_thz, phase_diff_sample_air_raw)
        plt.xlabel('f (Hz)')
        plt.ylabel('phase diff (radians)')
        plt.gca().legend(('sample-air'))

        plt.subplot(224)
        plt.plot(freq_sample_thz, phase_diff_sample_air_raw_corrected)
        plt.xlabel('f (Hz)')
        plt.ylabel('phase diff (radians)')
        plt.gca().legend(('sample-air'))


        plt.tight_layout()
        plt.pause(0.2)


        '''****************** n, k, eps and sigma plots below *********'''
        '''figure 5'''
        plt.figure(5)
        plt.subplot(1,2,1)
        plt.title('n(f) plot of sample')
        #plt.scatter(freq_sample_thz, nsa)
        plt.plot(freq_sample_thz, n_sample)
        plt.xlabel('f (Hz)')
        plt.ylabel('n')
        plt.subplot(1,2,2)
        plt.title('k(f) plot of sample')
        #plt.scatter(freq_sample_thz, ksa)
        plt.plot(freq_sample_thz, k_sample)
        plt.xlabel('f (Hz)')
        plt.ylabel('k')
        plt.tight_layout()
        plt.pause(0.2)


        '''figure 6'''
        plt.figure(6)
        plt.subplot(2,2,1)
        plt.title('eps1(f) plot of sample')
        plt.plot(freq_sample_thz, eps1_sample)
        plt.xlabel('f (Hz)')
        plt.ylabel('eps1')
        plt.subplot(2,2,2)
        plt.title('eps2(f) plot of sample')
        plt.plot(freq_sample_thz, eps2_sample)
        plt.xlabel('f (Hz)')
        plt.ylabel('eps2')
        plt.tight_layout()
        plt.subplot(2,2,3)
        plt.title('sigma(f) plot of sample')
        #plt.scatter(freq_sample_thz, sigma1_sample_est)
        plt.plot(freq_sample_thz, sigma1_sample)
        plt.xlabel('f (Hz)')
        plt.ylabel('sigma1 (S/m)')
        plt.tight_layout()
        plt.subplot(2,2,4)
        plt.title('sigma2(f) plot of sample')
        plt.plot(freq_sample_thz, sigma2_sample)
        plt.xlabel('f (Hz)')
        plt.ylabel('sigma2 (S/m)')
        plt.tight_layout()
        plt.pause(0.2)
