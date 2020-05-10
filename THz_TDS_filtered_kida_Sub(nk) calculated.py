# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:00:58 2019

This script initially performs the de-noising of THz-TDS data.
The script first removes the staright line slope from air, film+sub & substrate
data by high pass filtering and then removes the background data collected with
blank sample holder. The script next generates a window centered around the max
peak of signal & then performs the FFT of raw data and the windowed data.
It finally saves the FFT of the data in a new folder called 'FFT' in PWD.
The script saves magnitude & phase of the raw and windowed data in separate
files. The script does not optimize the n, k of the substrate but feed the
calculted n, k of sub to optimise the n, k of the film using
Kida formalism PRB 62, R11966 and saves the data for the electrical params in
separate folders for the raw and windowed data in the the PWD

@Author: Dr. Shilpam Sharma
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
#path = r'D:\THz TDS Expts\Mo thin films\Analyses\Mo_D\20052019\ascii data'
#backgnd_path = r'D:\THz TDS Expts\Mo thin films\Analyses\Mo_D\17052019\ds\ds1.txt'
#backgnd_path = r'D:\THz TDS Expts\Mo thin films\Analyses\Mo_A\14052019\ds\ascii data\Brass with thz on_500.txt'
path = r'C:\Users\A\Desktop\Mo_A - Copy\FullASCIIdata'
currentDirectory = pathlib.Path(path)

'''########### de-noising constants/prams defined here #########'''

f_min = 0.16e12             # minimum relevent frequency for data truncation
f_max = 1.05e12              # maximum relevent frequency for data truncation
lowcut = 0.05e12            # low cutoff of band pass butterworth filter
highcut = 1.4e12            # high cutoff of band pass butterworth filter
order = 10                   # order of the High pass butterworth filter
win_pts = 4500              # no. of points in the window around signal
backgnd_remove = False       # background removal required
data_file_header = True     # data file contains header or not
bck_file_header = True     # background file contains header or not
identifier = 'Ref'          # identifier for the file containing E vs t data of air
sample = 'Mo_A'             #name of the sample in the E vs t data files
sub = 'Si'                 #name of the substrate in the E vs t data files
plotting = True            # plotting of different quantities
'''######### optimization constants defined here ###############'''
d_sub = 0.532e-3            # susbtrate thickness in mts
d_film = 22.419e-9          # film thickness in mts
#nk_sub0 = [3.2, 0]          #initial guess of the substrate n and k
nk_film0 = [100.0, 100.0]   #initial guess of the film n and k
methods = 'NM'              # NM for Nelder-Mead or BFGS for L-BFGS-B algol

data_fft_win = pd.DataFrame()
data_fft_raw = pd.DataFrame()

'''dumping parameters to a binary file for later use'''
denoise_param = []
denoise_param.append(['f_min', 'f_max', 'lowcut', 'highcut', 'order', 'win_pts', 'd_film','d_sub'])
denoise_param.append([f_min, f_max, lowcut, highcut, order, win_pts, d_film, d_sub])
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

if backgnd_remove == True:
    backgnd_path = r'C:\Users\HP\Desktop\Mo_A\14052019\ds\ascii data\Brass with thz on_500.txt'
    if bck_file_header == True:
        data_backgnd = pd.read_csv(backgnd_path, delimiter = "\t", header= 0)
    else:
        data_backgnd = pd.read_csv(backgnd_path, delimiter = "\t", header = None, usecols = [0, 1], names = ['Interval', 'Data Level'])

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

def fitPhaseStLine(freq, m, c):
    return m*freq+c

'''################# Data reading starts here ###################'''
#currentDirectory = pathlib.Path(path)
currentPattern = "*.txt"
for currentFile in currentDirectory.glob(currentPattern):
    work_dir, cur_file = os.path.split(currentFile)
    idf = cur_file.split("_")
    if idf[0] == identifier:
        for i in idf:
            if i.find('K')!= -1:    # find temperature info from file name
                temperature = i
                t = temperature.split('K')[0]
                if t.find('p')!= -1:
                    t = float(t.replace('p', '.'))
                else:
                    t = float(t)

        file_air = cur_file
        file_film = cur_file.replace(identifier, sample)
        file_sub = cur_file.replace(identifier, sub)
        path_air = os.path.join(currentDirectory, file_air)
        path_sub = os.path.join(currentDirectory, file_sub)
        path_film = os.path.join(currentDirectory, file_film)
        print(temperature, '\n')
        # read data files
        if data_file_header == True:
            ''' uncomment here if file contains column names as header'''
            data_film = pd.read_csv(path_film, delimiter = "\t", usecols = ['Interval', 'Data Level'])
            data_sub = pd.read_csv(path_sub, delimiter = "\t", usecols = ['Interval', 'Data Level'])
            data_air = pd.read_csv(path_air, delimiter = "\t", usecols = ['Interval', 'Data Level'])
        else:
            '''use below lines to read the files without header info'''
            data_film = pd.read_csv(path_film, delimiter = "\t", header = None, usecols = [0, 1],
                                    names = ['Interval', 'Data Level'])
            data_sub = pd.read_csv(path_sub, delimiter = "\t", header = None, usecols = [0, 1], names = ['Interval', 'Data Level'])
            data_air = pd.read_csv(path_air, delimiter = "\t", header = None, usecols = [0, 1], names = ['Interval', 'Data Level'])


        ''' ############### High pass filtering of the backgnd data starts here ######################'''
        if backgnd_remove == True:
            x_backgnd = data_backgnd['Interval']
            y_backgnd = data_backgnd['Data Level']
            fs = 1/x_backgnd[1]
            y_backgnd_hpf = butter_bandpass_filter(y_backgnd, lowcut, highcut, fs , order)

        '''################ High pass filtering of the air data starts here ##########################'''
        # E vs t slope correction for air data
        x_air = data_air['Interval']
        y_air = data_air['Data Level']
        fs = 1/x_air[1]
        y_air_hpf = butter_bandpass_filter(y_air, lowcut, highcut, fs, order)

        '''################ High pass filtering of the film+substrate data starts here ################'''
        # E vs t slope correction for film+substrate data
        x_film = data_film['Interval']
        y_film = data_film['Data Level']
        fs = 1/x_film[1]
        y_film_hpf = butter_bandpass_filter(y_film, lowcut, highcut, fs, order)

        ''' High pass filtering of the substrate data starts here'''
        # E vs t slope correction for substrate data
        x_sub = data_sub['Interval']
        y_sub = data_sub['Data Level']
        fs = 1/x_sub[1]
        y_sub_hpf = butter_bandpass_filter(y_sub, lowcut, highcut, fs, order)

        '''Background Corrections are applied below'''
        if backgnd_remove == True:
            y_air_corr = pd.Series(y_air_hpf - y_backgnd_hpf)       #convert back to Pandas series)
            # film+substrate backgnd correction
            y_film_corr = pd.Series(y_film_hpf - y_backgnd_hpf)
            # substrate backgnd correction
            y_sub_corr = pd.Series(y_sub_hpf - y_backgnd_hpf)
        else:
            y_air_corr = pd.Series(y_air_hpf)                       #convert back to Pandas series)
            y_film_corr = pd.Series(y_film_hpf)
            y_sub_corr = pd.Series(y_sub_hpf)


        '''Air FFT to begin'''
        #calculate FFT of the raw time series of air
        sp_air_raw = fftshift(fft(y_air_corr))
        freq_air_raw = fftshift(fftfreq(x_air.size, x_air[1]))
        mag_air_raw = np.abs(sp_air_raw)
        phase_air_raw = np.unwrap(np.angle(sp_air_raw))
        '''reduce the data for centering the maximum signal with window function'''
        y_air_reduced = y_air_corr[y_air_corr.idxmax()-win_pts:y_air_corr.idxmax()+win_pts]
        '''choose a window to peep'''
        #window_air_reduced = signal.blackmanharris(y_air_reduced.size, sym = True)
        #window_air_reduced = signal.hann(y_air_reduced.size, sym = True)
        window_air_reduced = signal.boxcar(y_air_reduced.size, sym = True)
        #window_air_reduced = signal.kaiser(y_air_reduced.size, beta = beta_air)

        '''zero padding of window function '''
        #reduced_length = y_air_corr.size-y_air_reduced.size
        window_air = np.pad(window_air_reduced, (y_air_corr.idxmax()-win_pts, y_air_corr.size-(y_air_corr.idxmax()+win_pts)), 'constant', constant_values = (0,0))
        ''' Zero padding over'''
        '''FFT of the zero padded series'''
        sp_air_win = fftshift(fft(y_air_corr*window_air))
        mag_air_win = np.abs(sp_air_win)
        phase_air_win = np.unwrap(np.angle(sp_air_win))

        '''Film+susbtrate FFT begins here'''
        #calculate FFT of the raw time series for film+substrate
        sp_film_raw = fftshift(fft(y_film_corr))
        freq_film_raw = fftshift(fftfreq(x_film.size, x_film[1]))
        mag_film_raw = np.abs(sp_film_raw)
        phase_film_raw = np.unwrap(np.angle(sp_film_raw))
        '''reduce the data for centering the maximum signal with window function'''
        y_film_reduced = y_film_corr[y_film_corr.idxmax()-win_pts:y_film_corr.idxmax()+win_pts]
        '''choose a window to peep'''
        #window_film_reduced = signal.blackmanharris(y_film_reduced.size, sym = True)
        #window_film_reduced = signal.hann(y_film_reduced.size, sym = True)
        window_film_reduced = signal.boxcar(y_film_reduced.size, sym = True)
        #window_film_reduced = signal.kaiser(y_film_reduced.size, beta = beta_f_s)

        '''zero padding of  window function '''
        window_film = np.pad(window_film_reduced, (y_film_corr.idxmax()-win_pts, y_film_corr.size-(y_film_corr.idxmax()+win_pts)), 'constant', constant_values = (0,0))
        ''' Zero padding over'''
        '''FFT of the zero padded series'''
        sp_film_win = fftshift(fft(y_film_corr*window_film))
        mag_film_win = np.abs(sp_film_win)
        phase_film_win = np.unwrap(np.angle(sp_film_win))

        '''Substrate FFT begin here'''
        #calculate FFT of the raw time series for substrate
        sp_sub_raw = fftshift(fft(y_sub_corr))
        freq_sub_raw = fftshift(fftfreq(x_sub.size, x_sub[1]))
        mag_sub_raw = np.abs(sp_sub_raw)
        phase_sub_raw = np.unwrap(np.angle(sp_sub_raw))
        '''reduce the data for centering the maximum signal with window function'''
        y_sub_reduced = y_sub_corr[y_sub_corr.idxmax()-win_pts:y_sub_corr.idxmax()+win_pts]
        '''choose a window to peep'''
        #window_sub_reduced = signal.blackmanharris(y_sub_reduced.size, sym = True)
        #window_sub_reduced = signal.hann(y_sub_reduced.size, sym = True)
        window_sub_reduced = signal.boxcar(y_sub_reduced.size, sym = True)
        #window_sub_reduced = signal.kaiser(y_sub_reduced.size, beta = beta_f_s)

        '''zero padding of time series and window function '''
        window_sub = np.pad(window_sub_reduced, (y_sub_corr.idxmax()-win_pts, y_sub_corr.size-(y_sub_corr.idxmax()+win_pts)), 'constant', constant_values = (0,0))
        ''' Zero padding over'''
        '''FFT of the zero padded series'''
        sp_sub_win = fftshift(fft(y_sub_corr*window_sub))
        mag_sub_win = np.abs(sp_sub_win)
        phase_sub_win = np.unwrap(np.angle(sp_sub_win))


        '''Truncate the results to the relevent frequency range of f_min THz to f_max THz'''
        '''for air'''
        index_f_max = np.where(freq_air_raw<=f_max)[0][-1]
        index_f_min = np.where(freq_air_raw>=f_min)[0][0]
        freq_air_thz = freq_air_raw[index_f_min:index_f_max]
        sp_air_raw_thz = sp_air_raw[index_f_min:index_f_max]
        mag_air_raw_thz = mag_air_raw[index_f_min:index_f_max]
        phase_air_raw_thz = phase_air_raw[index_f_min:index_f_max]
        mag_air_win_thz = mag_air_win[index_f_min:index_f_max]
        phase_air_win_thz = phase_air_win[index_f_min:index_f_max]

        ''' for film+substrate'''
        index_f_max = np.where(freq_film_raw<=f_max)[0][-1]
        index_f_min = np.where(freq_film_raw>=f_min)[0][0]
        freq_film_thz = freq_film_raw[index_f_min:index_f_max]
        sp_film_raw_thz = sp_film_raw[index_f_min:index_f_max]
        mag_film_raw_thz = mag_film_raw[index_f_min:index_f_max]
        phase_film_raw_thz = phase_film_raw[index_f_min:index_f_max]
        mag_film_win_thz = mag_film_win[index_f_min:index_f_max]
        phase_film_win_thz = phase_film_win[index_f_min:index_f_max]
        ''' for substrate'''
        index_f_max = np.where(freq_sub_raw<=f_max)[0][-1]
        index_f_min = np.where(freq_sub_raw>=f_min)[0][0]
        freq_sub_thz = freq_sub_raw[index_f_min:index_f_max]
        sp_sub_raw_thz = sp_sub_raw[index_f_min:index_f_max]
        mag_sub_raw_thz = mag_sub_raw[index_f_min:index_f_max]
        phase_sub_raw_thz = phase_sub_raw[index_f_min:index_f_max]
        mag_sub_win_thz = mag_sub_win[index_f_min:index_f_max]
        phase_sub_win_thz = phase_sub_win[index_f_min:index_f_max]

        '''Transmission of Substrate with respect to air and film+substrate with respect to substrate'''
        Trans_sub_air = mag_sub_win_thz/mag_air_win_thz
        Trans_sub_air_raw = mag_sub_raw_thz/mag_air_raw_thz
        Trans_film_sub = mag_film_win_thz/mag_sub_win_thz
        Trans_film_sub_raw = mag_film_raw_thz/mag_sub_raw_thz
        Trans_film_air = mag_film_win_thz/mag_air_win_thz
        Trans_film_air_raw = mag_film_raw_thz/mag_air_raw_thz

        phase_diff_sub_air_raw = phase_sub_raw_thz-phase_air_raw_thz
        phase_diff_film_sub_raw = phase_film_raw_thz-phase_sub_raw_thz
        #phase_diff_film_air_raw = phase_film_raw_thz-phase_air_raw_thz
        phase_diff_sub_air_win = phase_sub_win_thz-phase_air_win_thz
        phase_diff_film_sub_win = phase_film_win_thz-phase_sub_win_thz
        #phase_diff_film_air_win = phase_film_win_thz-phase_air_win_thz

        optParam_sub_air, pcov = opt.curve_fit(fitPhaseStLine, freq_sub_thz, phase_diff_sub_air_raw, method='trf', bounds=((-np.inf, -np.inf), (np.inf, np.inf)))
        phase_diff_sub_air_raw_corrected = phase_diff_sub_air_raw-optParam_sub_air[1]
        optParam_sub_air, pcov = opt.curve_fit(fitPhaseStLine, freq_sub_thz, phase_diff_sub_air_win, method='trf', bounds=((-np.inf, -np.inf), (np.inf, np.inf)))
        phase_diff_sub_air_win_corrected = phase_diff_sub_air_win-optParam_sub_air[1]
        optParam_film_sub, pcov = opt.curve_fit(fitPhaseStLine, freq_film_thz, phase_diff_film_sub_raw, method='trf', bounds=((-np.inf, -np.inf), (np.inf, np.inf)))
        phase_diff_film_sub_raw_corrected = phase_diff_sub_air_raw-optParam_film_sub[1]
        optParam_film_sub, pcov = opt.curve_fit(fitPhaseStLine, freq_film_thz, phase_diff_film_sub_win, method='trf', bounds=((-np.inf, -np.inf), (np.inf, np.inf)))
        phase_diff_film_sub_win_corrected = phase_diff_sub_air_win-optParam_film_sub[1]

        '''The initial guess of n and k for whole frequency range is calculated here'''
        nsa = np.abs(1-((c/(2*pi*freq_sub_thz*d_sub))*phase_diff_sub_air_win_corrected))
        ksa = np.log((4*nsa)/((nsa+1)*Trans_sub_air))*(c/(2*pi*freq_sub_thz*d_sub))



        for r in range(len(freq_air_thz)):
            data_fft_raw.loc[r, 'Freq (cm-1)'] = freq_air_thz[r]/2.998e10
            data_fft_raw.loc[r, 'Freq (Hz)'] = freq_air_thz[r]
            data_fft_raw.loc[r, 'Magnitude (air)'] = mag_air_raw_thz[r]
            data_fft_raw.loc[r, 'Phase (air)'] = phase_air_raw_thz[r]
            data_fft_raw.loc[r, 'Freq (sub)'] = freq_sub_thz[r]
            data_fft_raw.loc[r, 'Magnitude (sub)'] = mag_sub_raw_thz[r]
            data_fft_raw.loc[r, 'Phase (sub)'] = phase_sub_raw_thz[r]
            data_fft_raw.loc[r, 'Freq (film)'] = freq_film_thz[r]
            data_fft_raw.loc[r, 'Magnitude (film)'] = mag_film_raw_thz[r]
            data_fft_raw.loc[r, 'Phase (film)'] = phase_film_raw_thz[r]
            data_fft_raw.loc[r, 'Tr (sub_air)'] = Trans_sub_air_raw[r]
            data_fft_raw.loc[r, 'Tr (film_sub)'] = Trans_film_sub_raw[r]
            #data_fft_raw.loc[r, 'Tr (film_air)'] = Trans_film_air_raw[r]
            data_fft_raw.loc[r, 'dPh (sub_air)'] = phase_diff_sub_air_raw[r]
            data_fft_raw.loc[r, 'dPh (film_sub)'] = phase_diff_film_sub_raw[r]
            #data_fft_raw.loc[r, 'dPh (film_air)'] = phase_diff_film_air_raw[r]
            #data_fft_raw.loc[r, 'Tr (film_sub_norm)'] = Trans_film_sub_raw[r]/freq_air_thz[r]

            data_fft_win.loc[r, 'Freq (cm-1)'] = freq_air_thz[r]/2.998e10
            data_fft_win.loc[r, 'Freq (Hz)'] = freq_air_thz[r]
            data_fft_win.loc[r, 'Magnitude (air)'] = mag_air_win_thz[r]
            data_fft_win.loc[r, 'Phase (air)'] = phase_air_win_thz[r]
            data_fft_win.loc[r, 'Freq (sub)'] = freq_sub_thz[r]
            data_fft_win.loc[r, 'Magnitude (sub)'] = mag_sub_win_thz[r]
            data_fft_win.loc[r, 'Phase (sub)'] = phase_sub_win_thz[r]
            data_fft_win.loc[r, 'Freq (film)'] = freq_film_thz[r]
            data_fft_win.loc[r, 'Magnitude (film)'] = mag_film_win_thz[r]
            data_fft_win.loc[r, 'Phase (film)'] = phase_film_win_thz[r]
            data_fft_win.loc[r, 'Tr (sub_air)'] = Trans_sub_air[r]
            data_fft_win.loc[r, 'Tr (film_sub)'] = Trans_film_sub[r]
            #data_fft_win.loc[r, 'Tr (film_air)'] = Trans_film_air[r]
            data_fft_win.loc[r, 'dPh (sub_air)'] = phase_diff_sub_air_win[r]
            data_fft_win.loc[r, 'dPh (film_sub)'] = phase_diff_film_sub_win[r]
            #data_fft_win.loc[r, 'dPh (film_air)'] = phase_diff_film_air_win[r]
            #data_fft_win.loc[r, 'Tr (film_sub_norm)'] = Trans_film_sub[r]/freq_air_thz[r]
        temp_out = np.full(len(freq_air_thz), t)
        file_out_raw = "{filename}_{samp}_{ids}.{ext}".format(filename=temperature, samp = sample, ids='FFT_raw', ext=idf[-1].split('.')[-1])
        file_out_win = "{filename}_{samp}_{ids}.{ext}".format(filename=temperature, samp = sample, ids='FFT_win', ext=idf[-1].split('.')[-1])
        file_out_p_raw = "{filename}_{samp}_{ids}.{ext}".format(filename=temperature, samp = sample, ids='EleParams_raw', ext=idf[-1].split('.')[-1])
        file_out_p_win = "{filename}_{samp}_{ids}.{ext}".format(filename=temperature, samp = sample, ids='EleParams_win', ext=idf[-1].split('.')[-1])
        new_directory_raw = os.path.join(currentDirectory, 'FFT Data_raw')
        new_directory_win = os.path.join(currentDirectory, 'FFT Data_win')
        new_directory_p_raw = os.path.join(currentDirectory, 'Ele_params_raw')
        new_directory_p_win = os.path.join(currentDirectory, 'Ele_params_win')
        if not os.path.exists(new_directory_raw):
            os.makedirs(new_directory_raw)
        if not os.path.exists(new_directory_win):
            os.makedirs(new_directory_win)
        if not os.path.exists(new_directory_p_raw):
            os.makedirs(new_directory_p_raw)
        if not os.path.exists(new_directory_p_win):
            os.makedirs(new_directory_p_win)
        path_out_raw = os.path.join(currentDirectory, 'FFT Data_raw', file_out_raw)
        path_out_win = os.path.join(currentDirectory, 'FFT Data_win', file_out_win)
        data_fft_raw.to_csv(path_out_raw, index=False, sep='\t', header=data_fft_raw.columns.tolist())
        data_fft_win.to_csv(path_out_win, index=False, sep='\t', header=data_fft_win.columns.tolist())

        '''*********************************************************'''
        '''########## Optimisation of n, k starts here #############'''
        '''*********************************************************'''
        data_fft_name = [data_fft_raw, data_fft_win] # for running the optimization on both raw and windowed data
        dir_names = ['Ele_params_raw', 'Ele_params_win']    #give different directories for raw and windowed data
        file_names = [file_out_p_raw, file_out_p_win]
        for name in range(len(data_fft_name)):

            '''some makeup for the optimized data containers'''
            n_sub = np.empty(len(data_fft_name[name]['Freq (Hz)']))
            k_sub = np.empty(len(data_fft_name[name]['Freq (Hz)']))
            n_film = np.empty(len(data_fft_name[name]['Freq (Hz)']))
            k_film = np.empty(len(data_fft_name[name]['Freq (Hz)']))
            data_Ele_param = pd.DataFrame()

            '''definition of the transfer function of air to substrate'''
            def transfer_sub(f, nk_sub):
                N2 = nk_sub[0]-1j*nk_sub[1]
                t3 = (2*N2)/(1+N2)
                t3_p = 2/(N2+1)
                k0 = (2*pi*f)/c
                k2 = k0*N2
                transmission = (t3*t3_p)*np.exp(-1j*(k2-k0)*d_sub)
                return transmission

            """ cost function of air to substrate for minimization"""
            def error_fun_sub(nk_sub, f, tr_exp_sub):
                return np.linalg.norm(tr_exp_sub-transfer_sub(f, nk_sub))

            '''fitting of the experimental transfer function of the substrate to theoretical model'''
            def match_sub(f,tr_exp_sub, nk_sub0):
                s_opt = minimize(error_fun_sub, nk_sub0, args=(f, tr_exp_sub), method='Nelder-Mead', tol = 1e-8, options={'return_all':False, 'adaptive': True, 'disp': True, 'maxiter':1000})
                return s_opt

            '''definition of the transfer function of film'''
            def transfer_film(f, nk_film, nk_sub):
                N1 = nk_film[0]-1j*nk_film[1]
                N2 = nk_sub[0]-1j*nk_sub[1]
                t1 = 2/(N1+1)
                t2 = (2*N1)/(N1+N2)
                t3_p = 2/(N2+1)
                k0 = (2*pi*f)/c
                k1 = k0*N1
                r1 = (1-N1)/(1+N1)
                r2 = (N1-N2)/(N1+N2)
                transmission_film = ((t1*t2)/t3_p)*((np.exp(-1j*(k1-k0)*d_film))/(1+(r1*r2)*np.exp(-1j*2*k1*d_film)))
                return transmission_film

            """ cost function of substrate to film for minimization """
            def error_fun_film(f, nk_film, tr_exp_film, nk_sub):
                return np.linalg.norm(tr_exp_film-transfer_film(nk_film, f, nk_sub))

            '''fitting of the experimental transfer function of the film to theoretical model'''
            if methods == 'NM':
                def match_film(f, nk_film0, tr_exp_film, nk_sub):
                    f_opt = minimize(error_fun_film, nk_film0, args=(f, tr_exp_film, nk_sub), method='Nelder-Mead', tol = 1e-8, options={'return_all':True, 'adaptive': True, 'disp': True, 'maxiter':1000})
                    return f_opt    # optimal solutions, other solutions'''
            elif methods == 'BFGS':
                def match_film(f, nk_film0, tr_exp_film, nk_sub):
                    bound = ((0.0, 1000.0), (0.0, 10000.0))
                    f_opt = minimize(error_fun_film, nk_film0, args=(f, tr_exp_film, nk_sub), method='L-BFGS-B', tol = 1e-8, bounds= bound)
                    return f_opt    # optimal solutions, other solutions'''
            '''calculation of the experimental transmission of air-substrate and film-substrate'''
            #tr_exp_sub = Trans_sub_air*np.exp(1j*(phase_diff_sub_air_win))
            #tr_exp_film = Trans_film_sub*np.exp(1j*(phase_diff_film_sub_win))
            #tr_exp_sub = data_fft_name[name]['Tr (sub_air)']*np.exp(1j*(data_fft_name[name]['dPh (sub_air)']))
            tr_exp_film = data_fft_name[name]['Tr (film_sub)']*np.exp(1j*(data_fft_name[name]['dPh (film_sub)']))

            #'''do minimization of n & k of substrate and film for each frequency'''
            for i in range(len(data_fft_name[name]['Freq (Hz)'])):
                f = data_fft_name[name]['Freq (Hz)'][i]
                '''nk_sub0 = [nsa[i], ksa[i]]
                nk_opt_sub = match_sub(f, tr_exp_sub[i], nk_sub0) # substrate minimization
                n_sub[i], k_sub[i] = nk_opt_sub.x
                #nk_sub = nk_opt_sub.x              # feed optimized n and k of the substarte for film optimization'''
                nk_sub = [nsa[i], ksa[i]]           # feed direct calculated n and k of the substarte for film optimization
                nk_opt_film = match_film(f, nk_film0, tr_exp_film[i], np.abs(nk_sub)) # film minimization
                n_film[i], k_film[i] = np.abs(nk_opt_film.x)

            '''conductivity and dielctric constant calculations'''
            '''substrate'''
            '''eps1_sub = (n_sub**2)-(k_sub**2)
            eps2_sub = 2.0*(n_sub*k_sub)
            sigma1_sub = 2.0*epsilon_0*(2.0*pi*data_fft_name[name]['Freq (Hz)'])*(n_sub*k_sub)
            sigma2_sub = epsilon_0*(2.0*pi*data_fft_name[name]['Freq (Hz)'])*(1.0-(n_sub**2)+(k_sub**2))'''
            eps1_sub = (nsa**2)-(ksa**2)
            eps2_sub = 2.0*(nsa*ksa)
            sigma1_sub = 2.0*epsilon_0*(2.0*pi*data_fft_name[name]['Freq (Hz)'])*(nsa*ksa)
            sigma2_sub = epsilon_0*(2.0*pi*data_fft_name[name]['Freq (Hz)'])*(1.0-(nsa**2)+(ksa**2))
            '''film'''
            eps1_film = (n_film**2)-(k_film**2)
            eps2_film = 2.0*(n_film*k_film)
            sigma1_film = 2.0*epsilon_0*(2.0*pi*data_fft_name[name]['Freq (Hz)'])*(n_film*k_film)
            sigma2_film = epsilon_0*(2.0*pi*data_fft_name[name]['Freq (Hz)'])*(1.0-(n_film**2)+(k_film**2))

            data_Ele_param.insert(0,'Freq (Hz)', data_fft_name[name]['Freq (Hz)'])
            data_Ele_param.insert(1, 'Temperature (K)', temp_out)
            data_Ele_param.insert(2,'n film', np.abs(n_film))
            data_Ele_param.insert(3,'k film', np.abs(k_film))
            data_Ele_param.insert(4,'Sigma1_film (S/cm)', sigma1_film/100.0)    # divide by 100 for conversion to S/cm from S/m
            data_Ele_param.insert(5,'Sigma2_film (S/cm)', sigma2_film/100.0)
            data_Ele_param.insert(6,'Eps1_film', eps1_film)
            data_Ele_param.insert(7,'Eps2_film', eps2_film)
            data_Ele_param.insert(8,'n sub', np.abs(n_sub))
            data_Ele_param.insert(9,'k sub', np.abs(k_sub))
            data_Ele_param.insert(10,'Sigma1_sub (S/cm)', sigma1_sub/100.0)
            data_Ele_param.insert(11,'Sigma2_sub (S/cm)', sigma2_sub/100.0)
            data_Ele_param.insert(12,'Eps1_sub', eps1_sub)
            data_Ele_param.insert(13,'Eps2_sub', eps2_sub)

            path_out = os.path.join(currentDirectory, dir_names[name], file_names[name])
            data_Ele_param.to_csv(path_out, index=False, sep='\t', header=data_Ele_param.columns.tolist())

            '''**********### Data crunching ends here ###***************'''

        '''let's plot somethings for showoff purposes'''
        if plotting == True:
            plt.figure(num=1)
            plt.subplot(221)
            plt.title('E vs t plot')
            plt.xlabel('t (sec.)')
            plt.ylabel('E (a. u.)')
            plt.plot(x_air, y_air_corr)
            plt.plot(x_sub, y_sub_corr)
            plt.plot(x_film, y_film_corr)
            plt.gca().legend(('air', 'substrate', 'film+substrate'))
            plt.subplot(222)
            plt.title('FFT of Air signal')
            plt.xlabel('f (Hz)')
            plt.ylabel('magnitude (a. u.)')
            plt.semilogy(freq_air_thz, mag_air_raw_thz, 'o')
            plt.semilogy(freq_air_thz, mag_air_win_thz)
            plt.gca().legend(('raw data', 'window'))
            plt.subplot(223)
            plt.title('FFT of Film+Substrate signal')
            plt.xlabel('f (Hz)')
            plt.ylabel('magnitude (a. u.)')
            plt.semilogy(freq_film_thz, mag_film_raw_thz, 'o')
            plt.semilogy(freq_film_thz, mag_film_win_thz)
            plt.gca().legend(('raw data', 'window'))
            plt.subplot(224)
            plt.title('FFT of Substrate signal')
            plt.xlabel('f (Hz)')
            plt.ylabel('magnitude (a. u.)')
            plt.semilogy(freq_sub_thz, mag_sub_raw_thz, 'o')
            plt.semilogy(freq_sub_thz, mag_sub_win_thz)
            plt.gca().legend(('raw data', 'window'))
            plt.tight_layout()
            plt.pause(0.2)

            plt.figure(num=2)
            plt.subplot(121)
            plt.semilogy(freq_air_thz, mag_air_win_thz, freq_sub_thz, mag_sub_win_thz, freq_film_thz, mag_film_win_thz)
            plt.xlabel('f (Hz)')
            plt.ylabel('magnitude (a. u.)')
            plt.gca().legend(('air', 'substrate', 'film+substrate'))
            plt.subplot(122)
            plt.plot(freq_sub_thz, Trans_sub_air, freq_film_thz, Trans_film_sub)
            plt.xlabel('f (Hz)')
            plt.ylabel('transmission (a.u.)')
            plt.gca().legend(('substrate-air', 'film-substrate'))
            plt.tight_layout()
            plt.pause(0.2)

            plt.figure(num=3)
            plt.subplot(221)
            plt.title('phase of raw data')
            plt.plot(freq_air_thz, phase_air_raw_thz, freq_sub_thz, phase_sub_raw_thz, freq_film_thz, phase_film_raw_thz)
            plt.xlabel('f (Hz)')
            plt.ylabel('phase (radians)')
            plt.gca().legend(('air', 'substrate', 'film+substrate'))
            plt.subplot(222)
            plt.title('phase of windowed data')
            plt.plot(freq_air_thz, phase_air_win_thz, freq_sub_thz, phase_sub_win_thz, freq_film_thz, phase_film_win_thz)
            plt.xlabel('f (Hz)')
            plt.ylabel('phase (radians)')
            plt.gca().legend(('air', 'substrate', 'film+substrate'))
            plt.subplot(223)
            plt.plot(freq_sub_thz, phase_diff_sub_air_raw, freq_film_thz, phase_diff_film_sub_raw)
            plt.xlabel('f (Hz)')
            plt.ylabel('phase diff (radians)')
            plt.gca().legend(('substrate-air', 'film-substrate'))
            plt.subplot(224)
            plt.plot(freq_sub_thz, phase_diff_sub_air_win, freq_film_thz, phase_diff_film_sub_win)
            plt.xlabel('f (Hz)')
            plt.ylabel('phase diff (radians)')
            plt.gca().legend(('substrate-air', 'film-substrate'))
            plt.tight_layout()
            plt.pause(0.2)

            plt.figure(num=4)
            plt.subplot(311)
            plt.title('Air')
            plt.plot(x_air, y_air_corr/y_air_corr.max(), x_air, window_air)
            plt.xlabel('t (Sec.)')
            plt.ylabel('E (a. u.)')
            plt.subplot(312)
            plt.title('Substrate')
            plt.plot(x_sub, y_sub_corr/y_sub_corr.max(), x_sub, window_sub)
            plt.xlabel('t (Sec.)')
            plt.ylabel('E (a. u.)')
            plt.subplot(313)
            plt.title('Film + Substrate')
            plt.plot(x_film, y_film_corr/y_film_corr.max(), x_film, window_film)
            plt.xlabel('t (Sec.)')
            plt.ylabel('E (a. u.)')
            plt.tight_layout()
            plt.pause(0.2)

            '''****************** n, k, eps and sigma plots below *********'''
            '''figure 5'''
            plt.figure(5)
            plt.subplot(1,2,1)
            plt.title('n(f) plot of substrate')
            plt.plot(freq_sub_thz, nsa)
            #plt.plot(freq_sub_thz, n_sub)
            plt.xlabel('f (Hz)')
            plt.ylabel('n')
            plt.subplot(1,2,2)
            plt.title('k(f) plot of substrate')
            plt.plot(freq_sub_thz, ksa)
            #plt.plot(freq_sub_thz, k_sub)
            plt.xlabel('f (Hz)')
            plt.ylabel('k')
            plt.tight_layout()
            plt.pause(0.2)

            '''figure 6'''
            plt.figure(6)
            plt.subplot(1,2,1)
            plt.title('n(f) plot of film')
            plt.plot(freq_film_thz, np.abs(n_film))
            plt.xlabel('f (Hz)')
            plt.ylabel('n')
            plt.subplot(1,2,2)
            plt.title('k(f) plot of film')
            plt.plot(freq_film_thz, np.abs(k_film))
            plt.xlabel('f (Hz)')
            plt.ylabel('k')
            plt.tight_layout()
            plt.pause(0.2)

            '''figure 7'''
            plt.figure(7)
            plt.subplot(2,2,1)
            plt.title('eps1(f) plot of substrate')
            plt.plot(freq_sub_thz, eps1_sub)
            plt.xlabel('f (Hz)')
            plt.ylabel('eps1')
            plt.subplot(2,2,2)
            plt.title('eps2(f) plot of substrate')
            plt.plot(freq_sub_thz, eps2_sub)
            plt.xlabel('f (Hz)')
            plt.ylabel('eps2')
            plt.tight_layout()
            plt.subplot(2,2,3)
            plt.title('sigma(f) plot of substrate')
            plt.plot(freq_sub_thz, sigma1_sub)
            plt.xlabel('f (Hz)')
            plt.ylabel('sigma1 (S/m)')
            plt.tight_layout()
            plt.subplot(2,2,4)
            plt.title('sigma2(f) plot of substrate')
            plt.plot(freq_sub_thz, sigma2_sub)
            plt.xlabel('f (Hz)')
            plt.ylabel('sigma2 (S/m)')
            plt.tight_layout()
            plt.pause(0.2)

            '''figure 8'''
            plt.figure(8)
            plt.subplot(2,2,1)
            plt.title('eps1(f) plot of film')
            plt.plot(freq_film_thz, eps1_film)
            plt.xlabel('f (Hz)')
            plt.ylabel('eps1')
            plt.subplot(2,2,2)
            plt.title('eps2(f) plot of film')
            plt.plot(freq_film_thz, eps2_film)
            plt.xlabel('f (Hz)')
            plt.ylabel('eps2')
            plt.tight_layout()
            plt.subplot(2,2,3)
            plt.title('sigma(f) plot of film')
            plt.plot(freq_film_thz, sigma1_film)
            plt.xlabel('f (Hz)')
            plt.ylabel('sigma1 (S/m)')
            plt.tight_layout()
            plt.subplot(2,2,4)
            plt.title('sigma2(f) plot of film')
            plt.plot(freq_film_thz, sigma2_film)
            plt.xlabel('f (Hz)')
            plt.ylabel('sigma2 (S/m)')
            plt.tight_layout()
            plt.pause(0.2)
            #plt.draw()

