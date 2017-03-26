"""
Some utility functions
"""

def plot_quiver(pa,s=40,figsize=(16,9)):
    """
    Given PA map, plot unit-length directionless vector field
    """
    plt.figure(figsize=figsize)
    X, Y = np.meshgrid(np.arange(pa.shape[1]),np.arange(pa.shape[0]))
    vx = np.cos(np.deg2rad(pa))
    vy = np.sin(np.deg2rad(pa))
    
    plt.quiver(X, Y, vx, vy, scale=s, headwidth=0)
    
    return vx,vy


def rdm_data_v(data,std=10):
    """
    Adding Gaussian PA uncertainties to velocity field.
    Default std = 10 deg 
    """
    unc = np.random.normal(scale=std,size=data.shape)
    return data + unc

def rdm_data_p(data,std=1e-5):
    """
    Adding Gaussian uncertainties to Stokes Q, U field.
    Default std = 10^-5, appropriate for the reported value for NGC2068

    Input
    - data : tuple of numpy arrays
      Stokes Q and U
    """
    Q, U = data
    uncQ = np.random.normal(scale=std,size=Q.shape)
    uncU = np.random.normal(scale=std,size=U.shape)
    return Q+uncQ, U+uncU


def pa2vxvy(pa):
    """
    Convert PA (deg) to velocity components in x and y directions
    """
    vx = np.cos(np.deg2rad(pa))
    vy = np.sin(np.deg2rad(pa))
    return vx, vy
