"""
Read the fits file
"""
class dataset(object):
    """
    Handling fits datasets. 
    Can be PA map (e.g. velocity flied)
    or Stokes Q, U maps (e.g. polarization)
    """
    def __init__(self,data_path,vg=False):
        self.data_path = data_path
        self.vg = vg
    
    def get_data(self):
        with get_readable_fileobj(self.data_path, cache=True) as e:
            fitsfile = fits.open(e)
            d        = fitsfile[0].data
            header   = fitsfile[0].header
        
        if self.vg==True:
            ## remove boundaries
            #  should always do this to numpy generated grad field
            return d[1:-1,1:-1]
        else: 
            if len(d.shape) > 2:
                ## Q, U
                return d[0]
            else:
                return d
