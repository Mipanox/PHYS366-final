"""
Methods for generating composite curl-noises from Perlin noise functions
"""

def perlin(x,y,seed=1240):
    """
    Generate Perlin gradient noises.
    See e.g., the blog post by Flafla2:
     http://flafla2.github.io/2014/08/09/perlinnoise.html

    Inputs:
    - x, y : each numpy ndarray (here 2D)
      The meshgrid for coordinates

    Parameters:
    - seed : integer (32 bit)
      The seed for the PRNG. Fix this if you want to reproduce the noise.
      Defaults to 1240

    Returns:
    - The perlin gradient noise, numpy ndarray
    """
    #-- permutation table
    ## important! store the seed for PRNG!
    np.random.seed(seed)
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    
    #-- coordinates of the top-left. The 'integral' grids
    xi = x.astype(int)
    yi = y.astype(int)

    #-- the relative internal coordinates
    xf = x - xi
    yf = y - yi
    
    #-- fade factors for smoothing
    u = fade(xf)
    v = fade(yf)
    
    #-- noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    
    #-- combine noises
    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u)
    
    return lerp(x1,x2,v) + 1e-15 # avoid zero

##############
#-for perlin-#

def lerp(a,b,x):
    """
    Linear interpolating the target point value
    """
    return a + x * (b-a)

def fade(t):
    """
    The standard fade function
    """
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    """
    The dot product average of the 'distance' of the point
    """
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

#-          -#
##############

def curl_noise(size,freq,seed,eps=1e-4):
    """
    Generate curl-noise based on a stream function defined by the
    Perlin noise by the 'perlin' method.
     See also Bridson+2007

    Inputs:
    - size : tuple of integers
      The size of the target curl-noise

    Parameters:
    - freq : float (conventionally integers)
      The 'octave' scale of the noise field.
      Namely the size of the coherent vortices.
    - seed : integer (32 bit). See 'perlin'
      Better no default to avoid carelessly assign fixed noises...
    - eps : float
      Precision for the finite difference differentiation.
      Can be very small (but larger than float precision of course),
      thanks to the 'continuity' of Perlin noises

    Returns:
    - The scaled curl-noise vector field (tuple of numpy ndarrays)
    """
    ## as tested, freq should not get below 1
    ## otherwise there's "repeated" pattern
    x,y = np.meshgrid(np.linspace(0,size[0]/2/freq,size[0]),
                      np.linspace(0,size[1]/2/freq,size[1]))

    ## Generate stream noise function
    stream = perlin(x,y,seed=seed)
    rot_gx =  (perlin(x,y+eps,seed=seed) - perlin(x,y-eps,seed=seed)) / 2
    rot_gy = -(perlin(x+eps,y,seed=seed) - perlin(x-eps,y,seed=seed)) / 2

    ## normalization factor 
    rot_norm = np.sqrt(rot_gx**2+rot_gy**2)
    return rot_gx/rot_norm,rot_gy/rot_norm # unit length

def per_sum(size,weights,seed,freq=[1,4,16]):
    """
    Combine three curl-noises of different scales with weights.
    The three scales correspond to freq=1,4,16 for the 'curl_noise'

    Inputs:
    - size : tuple of integers
      The size of the target curl-noise
    - weights: list or numpy array of length 3
      The weights of the three scales.
      Should sum up to unity (can be released)

    Parameters:
    - seed : see 'curl_noise' or 'perlin'
    - freq : list
      Default to [1,4,16]. If changing the frequencies (octaves),
      the "weights" should follow

    Returns:
    - per_sum : numpy ndarray (here 2D)
      The resulting PA of the composite curl-noise. 
    """
    weights = np.asarray(weights)

    ######
    #- uncomment this if the weights do not sum to one
    #tot = np.sum(weights)
    ######
    
    per_sum_x,per_sum_y = 0, 0 ## initialize

    ## Add the curl-noises vectorially
    for (i,w) in enumerate(weights):
        rot = curl_noise(size=size,freq=freq[i],seed=seed)
        per_sum_x += w * rot[0]
        per_sum_y += w * rot[1]

    ######
    #- uncomment this if the weights do not sum to one
    #per_sum_x /= tot
    #per_sum_y /= tot
    ######
    
    per_sum = np.rad2deg(np.arctan2(per_sum_y,per_sum_x))
    return per_sum # return is PA in deg!
