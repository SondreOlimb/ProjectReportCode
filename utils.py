from matplotlib import cm
import pandas as pd
import numpy as np
import seaborn as sns; 
from matplotlib.colors import  LogNorm
from scipy.signal import butter, lfilter, freqz,detrend
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy import ndimage
from PIL import Image, ImageFilter
import cv2
from operator import itemgetter



B = 194e6 # Sweep Bandwidth 
f_s =191.0153846*2
T =  1/B # Sweep Time
N = 256 # Sample Length
CPI =N*T # Coherent Processing Interval (CPI)

sample_bandwidth  = 191.0153846e6
sample_frek = 2*sample_bandwidth # sample frekuenzy
frame_rate =50*10**(-3) # frame rate
sample_per_frame = 260*257 # sample pr frame
c_t = (1/sample_bandwidth ) # sample period
PRI = 1/sample_frek*256 # PRI¨
PRF = 1/PRI

c = constants.c
m_w= B/c_t
frequencies = np.arange(0, 256//2)*f_s/256

def freq_to_range(f):
    return f*c/(2*m_w)

#range = np.arange(0, 255,4)

labels = {
    "x_label":"",
    "y_label":"",
    "title": ""

}

def fft_and_plot(data, axis, fs=1,fft_size=256,plot=False,shift =False,dB=True, vmin=1,vmax=100, doppler =False,labels = labels , savefig = False,figname ="", mode = 1):
    data_fft = np.fft.fft(data, axis=axis,n=fft_size)
    #data_fft = cv2.GaussianBlur(np.real(data_fft) +1j*np.imag(data_fft), (3, 3),sigmaX=mode,sigmaY=mode)
    if shift:
        data_fft = np.fft.fftshift(data_fft, axes=axis)
    


       
    # else:
    #     data_fft_plot = np.abs(data_fft)
    freq = np.fft.fftfreq(fft_size, d=1/f_s)

    #print(freq_to_range(frequencies))
    #print(len(range))
    #print(len(range(0,200,200/256)))

    if plot:
        
  
        plt.figure(figsize=(10,10))
        rotated_img = ndimage.rotate(data_fft,90) # We rotate the image so the x axis is the velocity
       

        rms = 10*np.log10(np.sqrt(np.mean(np.abs(rotated_img[130:135,80:100])**2)))
        peak = 10*np.log10(np.abs(rotated_img[137,87]))
        snr = peak-rms
        print("Peak:",peak)
        print("Side loab:", 10*np.log10(np.abs(rotated_img[137,85])))
        print("RMS:",rms)
        print("SNR:",snr) 
        if(dB):
        
      
        
           rotated_img = 20*np.log10(np.abs(rotated_img))
       

        else:
            rotated_img = np.abs(rotated_img)  
        plt.imshow(rotated_img,cmap="plasma", vmin=vmin,vmax=vmax)
        
        
        plt.yticks(np.linspace(0,256,5),labels=np.round(np.linspace(255*0.785277,0,5)),size =20)
        plt.xticks(size =20)
        if(doppler):
            plt.xticks(np.linspace(0,256,7),labels=np.round(np.linspace(-0.127552440715*127,0.127552440715*127,7),2),size =20)
        cbar  = plt.colorbar()
        cbar.set_label('Mangnitude [dB]',fontdict = {'fontsize' : 20})
        cbar.ax.tick_params(labelsize=15) 
        plt.xlabel(labels["x_label"],fontdict = {'fontsize' : 20})
        plt.ylabel(labels["y_label"],fontdict = {'fontsize' : 20})
        plt.title(labels["title"],fontdict = {'fontsize' : 30})
        plt.grid(False)
        #plt.tight_layout()
        #plt.margins(0.5,0.5)



        #ax = sns.heatmap(data_fft_plot,norm =LogNorm(vmin=10000),cmap="plasma")
        #ax = sns.heatmap(data_fft_plot,norm =LogNorm(vmin=treshold),cmap="plasma",xticklabels=range)
        #ax.set_xticks(range)


    #plt.plot()
        if savefig:
            plt.savefig(f'plots/{figname}.svg',format="svg")


    return data_fft

    
def plot_3D(data,figname,zlim_min=40,zlim_max=120):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    import numpy as np
    plt.figure(figsize=(60,60))
    data_abs = 20*np.log10(np.abs(data))
    #data_abs = np.delete(data_abs,0,axis=1)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    


    # Make data.
    X = np.arange(0, 256, 1)
    Y = np.arange(0, 256, 1)
    X, Y = np.meshgrid(X, Y)
    data_abs[data_abs<zlim_min]= np.nan
   
    

    # Plot the surface.
    surf = ax.plot_surface(X, Y, data_abs, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, vmin=zlim_min,vmax=zlim_max)
    #ax.set_zlim(zlim_min, zlim_max)
    
    ax.set_xticks(np.linspace(0,256,5),labels=np.round(np.linspace(0,255*0.785277,5)),size =15)
    
        
    ax.set_yticks(np.linspace(0,256,5),labels=np.round(np.linspace(-0.127552440715*127,0.127552440715*127,5),1),size =15)
    
    plt.xlabel(labels["x_label"],fontdict = {'fontsize' : 20})
    plt.ylabel(labels["y_label"],fontdict = {'fontsize' : 20})
    ax.set_zlim(zlim_min,zlim_max+30)
    
    cbar = fig.colorbar(surf, ax=ax ,shrink=0.5, aspect=5)
    cbar.set_label('Mangnitude [dB]',fontdict = {'fontsize' : 20})
    cbar.ax.tick_params(labelsize=10) 
    plt.savefig(f'plots/{figname}.svg',format="svg")

    plt.show()
    


def MIT_filter(data,fs=f_s,highcut=42,order=2):
    nyq = 0.5 * fs
    
    high = highcut / nyq
    sos = signal.butter(order, 42, 'hp',  output='sos',fs=fs)
    filtered = signal.sosfilt(sos, data,axis=0)
    return filtered

def full_process(data,fs=1,highcut=0.1,order=5,fft_size=256,shift=False):
    filtered = MIT_filter(data,fs,highcut,order)
    ffted = fft_and_plot(filtered,0,fs,fft_size,shift=shift)
    return ffted


#Data formatmating
def data_formatting(data):
    data = data.reshape(256,256)
    data = np.transpose(data)
    return data


#CFAR

def P_avg(P,N):
    return P/N
def alpha(N,P_FA):
    return N*(P_FA**(-1/N)-1)

def estimated_teshold(alpha,P):
    return alpha*np.abs(P)
def window_estimator(x,training_cells,training_area):    
    
    P_total = np.sum(np.abs(x))
    P_center_square =np.sum(np.abs(x[training_cells:x.shape[0]-training_cells,training_cells:x.shape[1]-training_cells]))
    P_traning_cells = np.abs(P_total - P_center_square)
    
    return P_avg(P_traning_cells,training_area)

def CFAR_2D(data, guard_cells, training_cells, PFA,plot = False,iso_axis =False,saveFig=False,filename =""):
    """_summary_

    Args:
        data (_type_): _description_
        guard_cells (_type_): _description_
        training_cells (_type_): _description_
        threshold (_type_): _description_
    """
    idx_peaks = []
    treshold_map = np.zeros(data.shape)

    window_size = guard_cells + training_cells
    data = ndimage.rotate(data, 90)
    
    data_cfar = np.pad(data, window_size, mode='edge')
    

    window_area = (2*window_size+1)**2
    training_area = window_area - (2*window_size+1-2*training_cells)**2
    print("traning area",training_area)
    a = alpha(training_area, PFA)
    detections = []

    for i in range(256):
        for j in range(256):
            P_training = window_estimator(data_cfar[i:i+2*window_size+1,j:j+2*window_size+1],training_cells ,training_area)
            threshold = estimated_teshold(a,P_training)
            treshold_map[i,j] = threshold
            
            if(np.abs(data[i,j]) < threshold):
                data[i,j] = 1
            else:
                idx_peaks.append([i,j])
                detections.append({
                    "cords":(i,j),
                    "peak":np.abs(data[i,j]),
                    "noise_est":np.abs(P_training),
                    "SNR":10*np.log10(np.abs(data[i,j])/np.abs(P_training))
                    }
                 )
                
    
    if(plot):
        newlist = sorted(detections, key=itemgetter("SNR"),reverse=True)
        print("Total detections",len(newlist))
        for i in newlist:
            print(i,"\n")
        plt.figure(figsize=(20,15))
        #rotated_img = ndimage.rotate(np.abs(data),90) # We rotate the image so the x axis is the velocity
        rotated_img =20*np.log10(np.abs(data))
        plt.imshow(rotated_img,cmap="plasma")
        plt.grid(False)
        if(iso_axis):
            plt.yticks(np.linspace(0,256,9),labels=np.linspace(200,0,9))
        
            plt.xticks(np.linspace(0,256,7),labels=np.round(np.linspace(-16.1987,16.1987,7),2))
        cbar  = plt.colorbar()
        cbar.set_label('Mangnitude [dB]',fontdict = {'fontsize' : 20})
        plt.xlabel(labels["x_label"],fontdict = {'fontsize' : 20})
        plt.ylabel(labels["y_label"],fontdict = {'fontsize' : 20})
        plt.title("CFAR",fontdict = {'fontsize' : 30})
        if saveFig:
            plt.savefig(f'plots/{filename}.svg',format="svg")
        plt.show()

        #3dplot
        # plt.figure(figsize=(60,60))
        # data_abs = 20*np.log10(np.abs(data))
        #     #data_abs = np.delete(data_abs,0,axis=1)

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            
            


        #     # Make data.
        # X = np.arange(0, 256, 1)
        # Y = np.arange(0, 256, 1)
        # X, Y = np.meshgrid(X, Y)
        # #ata_abs[data_abs<zlim_min]= np.nan
        
            

        #     # Plot the surface.
        # surf = ax.plot_surface(X, Y, data_abs, cmap=cm.coolwarm,
        #                         linewidth=0, antialiased=False)
        #     #ax.set_zlim(zlim_min, zlim_max)
            
        # ax.set_xticks(np.linspace(0,256,5),labels=np.round(np.linspace(0,255*0.785277,5)),size =10)
            
                
        # ax.set_yticks(np.linspace(0,256,5),labels=np.round(np.linspace(-0.127552440715*127,0.127552440715*127,5),1),size =10)
        # ax.set_zlabel("Magnitude [dB]")
        # ax.set_ylabel("Velocity [knots]")
        # ax.set_xlabel("Range [m]")
        # #ax.set_zlim(zlim_min,zlim_max+30)
            
        # cbar = fig.colorbar(surf, ax=ax ,shrink=0.5, aspect=5)
        # cbar.set_label('Mangnitude [dB]',fontdict = {'fontsize' : 10})
        # cbar.ax.tick_params(labelsize=10) 
        # if saveFig:
        #     plt.savefig(f'plots/{filename}_3D.svg',format="svg")
        


        # #plt.show()
    
    
    
            
    return data, detections, treshold_map


def radar_detection_algorithem(data_I,data_Q,fs=1,highcut=0.1,order=5,fft_size=256,guard_cells=5,training_cells=15,PFA=0.0001,plot=False):
    #Step 1: remove DC
    data_I = detrend(data_I, axis=2)
    data_Q = detrend(data_Q, axis=2)

    #Step 2: apply windowing in fast time
    data = signal.windows.hann(256)*data_I[:,:,:] + signal.windows.hann(256)*1j*data_Q[:,:,:]

    #stap 3: take the mean of the data
    data = np.mean(data, axis=0)

    #Step 4: Fast time range compression
    range_compression = np.fft.fft(data, axis=1,n=fft_size)

    #Step 5: apply windowing in slow time
    for i in range(256):
        range_compression[:,i] = signal.windows.hann(256)*range_compression[:,i]
    
    #Step 6: MIT filter
    filtered = MIT_filter(range_compression,fs,highcut,order)

    #Step 7: Doppler procesing
    data_doppler_processing = np.fft.fft(filtered, axis=0,n=fft_size)
    data_doppler_processing = np.fft.fftshift(data_doppler_processing, axes=0)

    #Step 8: Remove arti
    data_doppler_processing[91:95,204:207] = 1
    data_doppler_processing[161:167,50:54] = 1

    #Step 9: CFAR
    cfar, idx_peaks, treshold_map = CFAR_2D(data_doppler_processing,guard_cells,training_cells,PFA)
    print(idx_peaks)

    if plot:
        plt.figure(figsize=(20,15))
        rotated_img = ndimage.rotate(cfar,90) # We rotate the image so the x axis is the velocity
        plt.imshow(rotated_img,cmap="plasma")
        
        plt.yticks(np.linspace(0,256,5),labels=np.linspace(200,0,5))
       
        plt.xticks(np.linspace(0,256,11),labels=np.round(np.linspace(-16.1987,16.1987,11),2))
        cbar  = plt.colorbar()
        cbar.set_label('Mangnitude [dB]',fontdict = {'fontsize' : 20})
        plt.xlabel(labels["x_label"],fontdict = {'fontsize' : 20})
        plt.ylabel(labels["y_label"],fontdict = {'fontsize' : 20})
        plt.title(labels["title"],fontdict = {'fontsize' : 30})
        plt.grid(False)

    return cfar, idx_peaks, treshold_map