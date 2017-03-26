"""
Used to plot out best-fit curves by assigning weights and so on.
"""

def fit_out(w,data,label='Best-fit',style='b-'):
    """
    Plot the total RAPS of given weights for the composite curl-noise.
    Used when reading off best-fit weights and plot for visual check.

    Inputs:
    - w : list or numpy array of length 3
      (or depending on frequencies, see 'per_sum' in 'noise_gen.py')
      The weights for the composing curl-noises
    - data : numpy ndarray or tuple of np.ndarray
      The data to be compared with.
      In degree (PA) or numbers (Q, U; tuple)

    Options:
    - label / style: see 'tot_spec' in 'computation.py'

    Returns:
    - No return, only plots
    """
    #-- generate spectra to plot. Same in MCMC walkers
    if len(data)==2:
        ## if given polarization Q U data
        ### adding noises. Can skip by setting std=0...
        rdm_dQ,rdm_dU = rdm_data_p(data)
        E1, B1, _, _ = QU_to_EB(rdm_dQ,rdm_dU)
        data_Tpsd = tot_spec(E1,B1,plot=True,label='Obs.',style='r-')
        size = data[0].shape
    else:
        ## if given PA of vector field
        rdm_d = rdm_data_v(data)
        data_x, data_y = pa2vxvy(rdm_d)
        E1, B1, _, _ = xy_to_EB(data_x,data_y)
        data_Tpsd = tot_spec(E1,B1,plot=True,label='Obs.',style='r-')
        size = data.shape
    
    #-- generate the free field model according the weights
    weights = []
    for i in range(len(w)-1):
        weights.append(w[i])
    weights.append(1-np.sum(weights)) ## total summing to unity

    ## composing...
    ff = per_sum(size=size,weights=weights,seed=1240)

    #-- obtain the spectrum for the 'model'
    ff_x, ff_y = pa2vxvy(ff)
    E1f, B1f, _, _ = xy_to_EB(ff_x,ff_y)

    ## plot 
    ff_Tpsd = tot_spec(E1f,B1f,plot=True,label=label,style=style)


#################
# Example usage #
#################

## Plot obs. v.s. median best-fit spectra
# plt.figure(figsize=(16,9))
# ww = np.array([0.31,0.37,0.32])
# fit_out(ww,data=n2068_p,label='Median best-fit',style='b--')
