"""
Essential codes for computing
- E / B decomposition in the flat-sky limit
- PSD and RAPS
- CSD
- chi-square distances between two PSDs
- Evidence
"""

def QU_to_EB(Q,U,conv=False,std=0.3,plot=False):
    """
    E/B decomposition given Stokes Q, U maps.
    By definition two-dimensional.
     Ref: Louis+2013, Tucci+2002

    Inputs
    - Q, U : numpy ndarrays
      The observed Stokes Q, U maps. NaN values can be present.

    Parameters:
    - conv : bool
      Convolve the maps with Gaussian kernels before FFT or not.
      (Using astropy codes to handle NaN values.)
      Default to False.
    
    - std : float
      Standard deviation for the Gaussian kernel.
      Default to 0.3 (of the map y-size). Not used if 'conv' is False

    - plot : bool
      Turn this on to plot the PSDs for the E/B-modes.
      The Kolmogorov power spectrum of turbulence will be plotted as well.

    Returns
    - E1 : numpy ndarray
      The two-dimensional E-mode distribution.
    - B1 : numpy ndarray
      The two-dimensional B-mode distribution.
    - Epsd : tuple of numpy arrays
      The frequencies and powers of the E-mode RAPS
    - Bpsd : tuple of numpy arrays
      The frequencies and powers of the B-mode RAPS
    """
    
    ## use astropy function to ignore nan values...
    qf = convolve_fft(Q,[[1]],interpolate_nan=True,return_fft=True,fft_pad=False) # no convolution
    uf = convolve_fft(U,[[1]],interpolate_nan=True,return_fft=True,fft_pad=False)
    
    if conv == True:
        std = std * Q.shape[1] # whatever, try 0.3 times the total size...
        g = Gaussian2DKernel(stddev=std)
        qf = convolve_fft(Q,g,interpolate_nan=True,return_fft=True,fft_pad=False)
        uf = convolve_fft(U,g,interpolate_nan=True,return_fft=True,fft_pad=False)

    #-- Fourier domain
    ## frequencies in l_x and l_y of the FFT
    fq_x,fq_y = np.meshgrid(np.fft.fftfreq(qf.shape[1]),np.fft.fftfreq(qf.shape[0]))
    
    ## phi_l : the angle between the vector l and the l_x axis
    phi_l = np.arctan2(fq_y,fq_x)
    phi_c = 2 * phi_l
    
    ef =  qf * np.cos(phi_c) + uf * np.sin(phi_c)
    bf = -qf * np.sin(phi_c) + uf * np.cos(phi_c)
    
    #-- IFT
    E1 = ifft2(ef)
    E2 = E1 - np.mean(E1) # get rid of the DC component
    Epsd = PSD2(E2, oned=True)
    B1 = ifft2(bf)
    B2 = B1 - np.mean(B1)
    Bpsd = PSD2(B2, oned=True)
    
    ## normalization to total power of 1
    norm_E = np.trapz(Epsd[1],Epsd[0])
    norm_B = np.trapz(Bpsd[1],Bpsd[0])
    
    Epsd[1], Bpsd[1] = Epsd[1]/norm_E, Bpsd[1]/norm_B

    ## plotting
    if plot==True:
        plt.figure(figsize=(9,9))
        plt.plot(Epsd[0],Epsd[1],'b',label='E-mode')
        plt.plot(Bpsd[0],Bpsd[1],'r',label='B-mode')
        plt.yscale('log'); plt.xscale('log'); 
        plt.xlabel('Spectral Frequency', fontsize=24)
        plt.ylabel('Radial Power', fontsize=24)
        plt.xticks(fontsize=20); plt.yticks(fontsize=20)
        plt.ylim(1e-4,100); plt.xlim(0.01,1)

        ## Kolmogorov spectrum
        xx = np.linspace(1e-3,1,1000)
        ko = xx**(-5/3)
        kn = ko/np.trapz(ko,xx)
        plt.plot(xx,kn,'g--',label='Kolmogorov -5/3')
        plt.legend(fontsize=24)
    
    return E1, B1, Epsd, Bpsd

def xy_to_EB(cx,cy,plot=False):
    """
    Compute E/B decomposition given x and y components of a vector field.

    Inputs
    - cx : numpy ndarray
      X component of the vector field
    - cy : numpy ndarray
      Y component of the vector field

    Options:
    - plot : bool
      See descriptions in 'QU_to_EB'

    Returns:
      See descriptions in 'QU_to_EB'
    """
    psi = np.arctan2(cy,cx) # use 'arctan2' to avoid division by zero
    Q = np.cos(2*psi)
    U = np.sin(2*psi)
    
    E1, B1, Epsd, Bpsd = QU_to_EB(Q,U,plot=plot)
    
    return E1, B1, Epsd, Bpsd

def tot_spec(E1,B1,plot=False,label='Total power',style='b'):
    """
    Total (combined) power spectrum (radial) of the E/B-modes

    Inputs:
    - E1 : numpy ndarray
      The E-mode component
    - B1 : numpy ndarray
      The B-mode component

    Options:
    - plot : bool
      See descriptions in 'QU_to_EB'
    - label : string
      Label of the spectrum. Default to "Total power"
    - style : string
      Plotting option for matplotlib pyplot lines
      Default to "b"

    Returns:
    - Tot : numpy ndarray
      Total 2D power spectral density

    - Tpsd : numpy array
      The RAPS of 'Tot'
    """
    ## remove the DC components
    E2 = E1 - np.mean(E1)
    B2 = B1 - np.mean(B1)
    Tot = E2 + B2 ## FT is linear
    Tpsd = PSD2(Tot, oned=True)
    
    ## normalization to total integrated power of unity
    Tpsd[1] = Tpsd[1]/np.trapz(Tpsd[1],Tpsd[0])
    
    if plot==True:
        #plt.figure(figsize=(9,9))
        plt.plot(Tpsd[0],Tpsd[1],style,label=label)
        plt.yscale('log'); plt.xscale('log'); 
        plt.xlabel('Spectral Frequency', fontsize=24)
        plt.ylabel('Radial Power', fontsize=24)
        plt.xticks(fontsize=20); plt.yticks(fontsize=20)

        ## Kolmogorov spectrum for comparison
        plt.ylim(1e-4,100); plt.xlim(0.01,1)
        xx = np.linspace(1e-3,1,1000)
        ko = xx**(-5/3)
        kn = ko/np.trapz(ko,xx)
        plt.plot(xx,kn,'g--',label='Kolmogorov -5/3')
        plt.legend(fontsize=24)
    
    return Tot, Tpsd


########
def cross(psd1,psd2):
    """
    Calculate the norm of the CSD of two PSDs.

    Inputs:
    - psd1 / psd2: numpy array
      The RAPS of a signal

    Returns:
    - The integral of the CSD of the input signals
    """
    ## obtain the 'time series'
    ff1, ff2 = abs(ifft(psd1[1])), abs(ifft(psd2[1]))
    f, Pxy = signal.csd(ff1, ff2, return_onesided=True)

    ## discrete integration
    return np.trapz(np.abs(Pxy)[2:],f[2:])

def chi_sq(psd1,psd2):
    """
    Chi-square distances between two curves,
    here the RAPS of the vector fields

    Inputs:
    - psd1 / psd2: numpy array
      The RAPS of a signal

    Returns:
    - The total chi-square distance between the two inputs
    """
    ln1,ln2 = len(psd1[1]),len(psd2[1])
    if ln1 > ln2:
        ## interpolate so that the two signals have the same length
        psd2_ = np.interp(psd1[0],psd2[0],psd2[1]) 
        psd1_ = psd1[1]
    else:
        psd1_ = np.interp(psd2[0],psd1[0],psd1[1]) 
        psd2_ = psd2[1]

    ## discrete integration
    return np.trapz((psd1_-psd2_)**2,psd1[0])


############
# Evidence #
data_p,data_v = [],[]
def evid(data_p=data_p,data_v=data_v,N_samples=2000):
    """
    Calculate the 'evidences' of models:
    (1) Frequentist: randomly "observing" vector & B-field many times
        measure how well they match
    (2) Bayesian evidence from sampling prior weights in [0,1]

    Inputs:
    - data_p : the Stokes Q, U maps
    - data_v : the vector field PA map

    Paramaters:
    - N_samples: integer
      How many priors / random fields to draw?

    Returns:
    - evd_field : float
      Evidence of P(obs_B|M) for observed B-field configuration
    
    - evd_vb : float
      Evidence for the similarity between the two fields
    
    - evd_f_max : float
      The maximum likelihood of the model

    - evd_v_max : float
      Maximum likelihood for the randomly "observed" two fields
    """
    ## initialize
    evd_field, evd_vb = 0., 0.  
    evd_f_max, evd_v_max = 0., 0.

    #- Generate uniform weights w1,w2,w3=U(0,1)
    ## randomly draw 2 numbers in [0,1]
    w1  = np.random.rand(N_samples)
    w2_ = np.random.rand(N_samples)
    w3_ = np.ones(N_samples)

    ## sort them and subtract to get the divisions
    stack = np.sort(np.vstack((w1,w2_,w3_)),axis=0)

    w2 = stack[1,:]-stack[0,:]
    w3 = stack[2,:]-stack[1,:]

    ## put them together
    w_prior = np.vstack((w1,w2,w3))

    #- Sum up the likelihoods...
    for i in range(N_samples):
        ### vector field
        rdm_d = rdm_data_v(data_v)
        data_x, data_y = pa2vxvy(rdm_d)
        E1v, B1v, _, _ = xy_to_EB(data_x,data_y)
        datav_Tpsd = tot_spec(E1v,B1v)
    
        ### B-field
        rdm_dQ,rdm_dU = rdm_data_p(data_p)
        Ep1, Bp1, _, _ = QU_to_EB(rdm_dQ,rdm_dU)
        datap_Tpsd = tot_spec(Ep1,Bp1)
        
        weights = w_prior[:,i]
        ff = per_sum(size=data_v.shape,weights=weights,
                     seed=np.random.randint(np.iinfo(np.int32).max))
    
        ff_x, ff_y = pa2vxvy(ff)
        E1f, B1f, _, _ = xy_to_EB(ff_x,ff_y)
        ff_Tpsd = tot_spec(E1f,B1f)
    
        evd_f_new = np.exp(-0.5*chi_sq(datap_Tpsd[1],ff_Tpsd[1]))
        evd_v_new = np.exp(-0.5*chi_sq(datap_Tpsd[1],datav_Tpsd[1]))
        
        evd_field += evd_f_new
        evd_vb    += evd_v_new

        ## update max.
        if evd_f_new > evd_f_max: evd_f_max = evd_f_new
        if evd_v_new > evd_v_max: evd_v_max = evd_v_new

        ## because this can be slow, print out status...
        if i%100==0:
            print 'sampling...: %d' %i
    
    return evd_field,evd_vb, evd_f_max, evd_v_maxn
