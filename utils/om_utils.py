
import numpy as np

from collections import OrderedDict

#TODO: Function to convert om dicts to smt ordering

# etc.

def get_om_dict(xnew, smt_map):


    # smt_map ('name', inds (int or np.ndarray))

    # from onerahub, om to smt bounds
    om_dvs = []

    for name, inds in smt_map.items():
       
        size = inds.shape[0]

        vals = np.zeros(size)

        for i in range(inds.shape[0]):
            vals[i] = xnew[inds[i]]
        
        om_dvs.append(name, vals)

    return om_dvs

def map_om_to_smt(self, dvs):
    """
    Map OM design var dict to a particular SMT ordering

    Do this once and pass the map as metadata to SMT-based components and 
    drivers

    Parameters
    ----------
    dvs : OrderedDict
        openmdao design variables

    Returns
    -------
    smt_map : dict
        indices for each design variable in a single vector
        
    xlimits : list
        list of ordered bounds
    """

    #dvs = self._designvars

    #{name:index}
    # establish this here

    smt_map = OrderedDict([(name, np.zeros(_get_size(meta), dtype=int))
                               for name, meta in dvs.items()])

    # from onerahub, om to smt bounds
    xlimits = []

    count = 0
    for name, meta in dvs.items():

        size = meta["size"]
        meta_low = meta["lower"]
        meta_high = meta["upper"]

        for j in range(size):
            if isinstance(meta_low, np.ndarray):
                p_low = meta_low[j]
            else:
                p_low = meta_low

            if isinstance(meta_high, np.ndarray):
                p_high = meta_high[j]
            else:
                p_high = meta_high

            xlimits.append((p_low, p_high))

            smt_map[name][j] = count
            count += 1

            

    return smt_map, xlimits


def get_om_design_size(dv_settings):
    ndesvar = 0
    for name, meta in dv_settings.items():
        size = meta['global_size'] if meta['distributed'] else meta['size']
        ndesvar += size
    return ndesvar

# Assume items in OM DV dict originate from OrderedDict
def om_dict_to_flat_array(dct, dv_settings, ndesvar):
    desvar_array = np.empty(ndesvar)
    i = 0
    for name, meta in dv_settings.items():
        size = meta['global_size'] if meta['distributed'] else meta['size']
        desvar_array[i:i + size] = dct[name]
        i += size
    return desvar_array

def _get_size(dct):
    # Returns global size of the variable if it is distributed, size otherwise.
    return dct['global_size'] if dct['distributed'] else dct['size']