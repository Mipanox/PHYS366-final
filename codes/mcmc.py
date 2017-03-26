"""
MCMC setup
"""

# Prior - uniform withn [0,1]
def log_prior(weights):
    if np.all(weights > 0) * (1-np.sum(weights)) > 0: 
        # if all positive and smaller than 1
        return 0.
    else:
        return -np.inf



# Posterior

##############
# chi-square #

### full : "noisy"
def log_post_v_chi(djunk=None):
    """
    log posterior for velocity gradient field.
    Likelihood defined by the chi-square distance.
    """
    ## generate spectra...
    ### data
    rdm_d = rdm_data_v(data_v)
    data_x, data_y = pa2vxvy(rdm_d)
    
    E1, B1, _, _ = xy_to_EB(data_x,data_y)
    data_Tpsd = tot_spec(E1,B1)
    
    ### free field
    weights = []
    for i in range(2):
        weights.append(w[i]())
    weights.append(1-np.sum(weights)) ## total summing to unity
    ff = per_sum(size=data_v.shape,weights=weights,
                 seed=np.random.randint(np.iinfo(np.int32).max))
    
    ff_x, ff_y = pa2vxvy(ff)
    E1, B1, _, _ = xy_to_EB(ff_x,ff_y)
    ff_Tpsd = tot_spec(E1,B1)
    
    chi = chi_sq(data_Tpsd[1],ff_Tpsd[1])
    
    return -0.5 * chi + log_prior(np.array([w[0](),w[1]()]))

def log_post_p_chi(djunk=None):
    """
    log posterior for B-field
    Likelihood defined by the chi-square distance.
    """
    ## generate spectra...
    ### data
    rdm_dQ, rdm_dU = rdm_data_p(data_p)
    
    E1, B1, _, _ = QU_to_EB(rdm_dQ,rdm_dU)
    data_Tpsd = tot_spec(E1,B1)
    
    ### free field
    weights = []
    for i in range(2):
        weights.append(w[i]())
    weights.append(1-np.sum(weights)) ## total summing to unity
    ff = per_sum(size=data_p[0].shape,weights=weights,
                 seed=np.random.randint(np.iinfo(np.int32).max))
    
    ff_x, ff_y = pa2vxvy(ff)
    E1, B1, _, _ = xy_to_EB(ff_x,ff_y)
    ff_Tpsd = tot_spec(E1,B1)
    
    chi = chi_sq(data_Tpsd[1],ff_Tpsd[1])
    
    return -0.5 * chi + log_prior(np.array([w[0](),w[1]()]))


### "noisless"
def log_post_v_chi_noiseless(djunk=None):
    """
    velocity field, noiseless
    """
    ## generate spectra...
    ### data
    data_x, data_y = pa2vxvy(data_v)
    
    E1, B1, _, _ = xy_to_EB(data_x,data_y)
    data_Tpsd = tot_spec(E1,B1)
    
    ### free field
    weights = []
    for i in range(2):
        weights.append(w[i]())
    weights.append(1-np.sum(weights)) ## total summing to unity
    ff = per_sum(size=data_v.shape,weights=weights,
                 seed=np.random.randint(np.iinfo(np.int32).max))
    
    ff_x, ff_y = pa2vxvy(ff)
    E1, B1, _, _ = xy_to_EB(ff_x,ff_y)
    ff_Tpsd = tot_spec(E1,B1)
    
    chi = chi_sq(data_Tpsd[1],ff_Tpsd[1])
    
    return -0.5 * chi + log_prior(np.array([w[0](),w[1]()]))

def log_post_p_chi_noiseless(djunk=None):
    """
    B-field, noiseless
    """
    ## generate spectra...
    ### data    
    E1, B1, _, _ = QU_to_EB(data_p[0],data_p[1])
    data_Tpsd = tot_spec(E1,B1)
    
    ### free field
    weights = []
    for i in range(2):
        weights.append(w[i]())
    weights.append(1-np.sum(weights)) ## total summing to unity
    ff = per_sum(size=data_p[0].shape,weights=weights,
                 seed=np.random.randint(np.iinfo(np.int32).max))
    
    ff_x, ff_y = pa2vxvy(ff)
    E1, B1, _, _ = xy_to_EB(ff_x,ff_y)
    ff_Tpsd = tot_spec(E1,B1)
    
    chi = chi_sq(data_Tpsd[1],ff_Tpsd[1])
    
    return -0.5 * chi + log_prior(np.array([w[0](),w[1]()]))

### "noisefree"
def log_post_v_chi_fixseed(djunk=None):
    """
    for velocity field with fixed model seed
    This will fit "the" particular curl_noise with different weightings.
    Strongly restrictive! Use with caution!
    """
    ## generate spectra...
    ### data
    rdm_d = rdm_data_v(data_v)
    data_x, data_y = pa2vxvy(rdm_d)
    
    E1, B1, _, _ = xy_to_EB(data_x,data_y)
    data_Tpsd = tot_spec(E1,B1)
    
    ### free field
    weights = []
    for i in range(2):
        weights.append(w[i]())
    weights.append(1-np.sum(weights)) ## total summing to unity
    ff = per_sum(size=data_v.shape,weights=weights,seed=1240)
    
    ff_x, ff_y = pa2vxvy(ff)
    E1, B1, _, _ = xy_to_EB(ff_x,ff_y)
    ff_Tpsd = tot_spec(E1,B1)
    
    chi = chi_sq(data_Tpsd[1],ff_Tpsd[1])
    
    return -0.5 * chi + log_prior(np.array([w[0](),w[1]()]))

def log_post_p_chi_fixseed(djunk=None):
    """ for B-field, fixed seed. See above for v """
    ## generate spectra...
    ### data
    rdm_dQ, rdm_dU = rdm_data_p(data_p)
    
    E1, B1, _, _ = QU_to_EB(rdm_dQ,rdm_dU)
    data_Tpsd = tot_spec(E1,B1)
    
    ### free field
    weights = []
    for i in range(2):
        weights.append(w[i]())
    weights.append(1-np.sum(weights)) ## total summing to unity
    ff = per_sum(size=data_p[0].shape,weights=weights,seed=1240)
    
    ff_x, ff_y = pa2vxvy(ff)
    E1, B1, _, _ = xy_to_EB(ff_x,ff_y)
    ff_Tpsd = tot_spec(E1,B1)
    
    chi = chi_sq(data_Tpsd[1],ff_Tpsd[1])
    
    return -0.5 * chi + log_prior(np.array([w[0](),w[1]()]))
    
###########
# CSD sum #

### only "noisy"
def log_post_v_csd(djunk=None):
    """
    log posterior for velocity gradient field.
    Likelihood defined by the integral of the CSD
    """
    ## generate spectra...
    ### data
    rdm_d = rdm_data_v(data_v)
    data_x, data_y = pa2vxvy(rdm_d)
    
    E1, B1, _, _ = xy_to_EB(data_x,data_y)
    data_Tpsd = tot_spec(E1,B1)
    
    ### free field
    weights = []
    for i in range(2):
        weights.append(w[i]())
    weights.append(1-np.sum(weights)) ## total summing to unity
    ff = per_sum(size=data_v.shape,weights=weights,
                 seed=np.random.randint(np.iinfo(np.int32).max))
    
    ff_x, ff_y = pa2vxvy(ff)
    E1, B1, _, _ = xy_to_EB(ff_x,ff_y)
    ff_Tpsd = tot_spec(E1,B1)
    
    ## csd
    csd = cross(data_Tpsd[1],ff_Tpsd[1])
    
    return np.log(csd) + log_prior(np.array([w[0](),w[1]()]))

def log_post_p_csd(djunk=None):
    """
    log posterior for B-field
    Likelihood defined by the integral of the CSD
    """
    ## generate spectra...
    ### data
    rdm_dQ, rdm_dU = rdm_data_p(data_p)
    
    E1, B1, _, _ = QU_to_EB(rdm_dQ,rdm_dU)
    data_Tpsd = tot_spec(E1,B1)
    
    ### free field
    weights = []
    for i in range(2):
        weights.append(w[i]())
    weights.append(1-np.sum(weights)) ## total summing to unity
    ff = per_sum(size=data_p[0].shape,weights=weights,
                 seed=np.random.randint(np.iinfo(np.int32).max))
    
    ff_x, ff_y = pa2vxvy(ff)
    E1, B1, _, _ = xy_to_EB(ff_x,ff_y)
    ff_Tpsd = tot_spec(E1,B1)
    
    ## csd
    csd = cross(data_Tpsd[1],ff_Tpsd[1])
    
    return np.log(csd) + log_prior(np.array([w[0](),w[1]()]))
    
