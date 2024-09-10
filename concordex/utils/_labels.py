import re
from warnings import warn

import numpy as np
import pandas as pd

from anndata import AnnData

class Labels:
    def __init__(self, names):
        
        if names is None:
           raise ValueError("No labels to search for. Must provide labels.")
        
        self._lookup = names

    @property
    def labeltype(self) -> str:
        if not hasattr(self, "_labeltype"):
            raise AttributeError("`labeltype` has not been set. Call `extract(adata)` first")
        
        return self._labeltype

    @property
    def labelnames(self) -> str | list:
        if not hasattr(self, '_labelnames'):
            raise AttributeError("`labelnames` has not been set. Call `extract(adata)` first.")
        
        return self._labelnames

    @property
    def values(self):
        if not hasattr(self, '_values'):
            raise AttributeError("`values` has not been set. Call `extract(adata)` first.")
        return self._values
    
    @property 
    def n_unique_labels(self):
        if not hasattr(self, '_labelshape'):
            raise AttributeError("Unable to determine number of labels. Call `extract(adata)` first.")
        return self._labelshape[1]

    @property
    def discretelabelscollapsed(self):
        if not hasattr(self, '_discretelabelscollapsed'):
            raise AttributeError("Ensure that labels are discrete and call `extract(adata)`.")
        return self._discretelabelscollapsed

    @property
    def discretelabelsunique(self):
        if not hasattr(self, '_discretelabelsunique'):
            raise AttributeError("Ensure that labels are discrete and call `extract(adata)`.")
        return self._discretelabelsunique

    @property
    def nbccolumns(self):
        if not hasattr(self, '_labelcolumns'):
            if not hasattr(self, '_values'):
                return []
            else:
                _nattr = self._values.shape[1]
                self._labelcolumns = [f"X_{i}" for i in range(_nattr)]
        
        return self._labelcolumns

    def extract(self, adata: AnnData):
        """
        Extract labels from adata.obs (or adata.obsm) and update 
        """
        if self._lookup is not None:

            obs_keys = adata.obs.keys()
            m = np.isin(obs_keys, self._lookup)
            
            if any(m):
                labels_sub = adata.obs[obs_keys[m]]
                types = [dt.name for dt in labels_sub.dtypes]
                
                self._labelnames = labels_sub.columns.tolist() 

                self._validate(types) 
                _values = labels_sub.values
                
                if self._labeltype == "discrete":
                    self._discretelabelscollapsed = self.collapse(_values)
                    _values_ohe = self.one_hot_encode(self._discretelabelscollapsed)

                    self._discretelabelsunique = _values_ohe.columns.tolist()
                    self._labelcolumns = _values_ohe.columns.tolist()

                    _values = _values_ohe.values

            else: 
                # Check if labels are in .obsm
                lookup_key = self._lookup
                if not isinstance(lookup_key, str):
                    lookup_key = self._lookup[0]
                    warn(
                        f"Looking for labels in `adata.obsm`. Only the first key, {lookup_key}, will be used.",
                        category=UserWarning
                    )
                
                if lookup_key in adata.obsm.keys():
                    self._labeltype = 'continuous'
                    _values = adata.obsm[lookup_key]
                    self._labelnames = lookup_key

                    # Keep track of colnames for NBC
                    _nattr = _values.shape[1]
                    self._labelcolumns = [f"{lookup_key}_{i}" for i in range(_nattr)]

                else:
                    raise KeyError(
                        f"{lookup_key} not found in `adata`"
                        )
            self._labelshape = (_values.shape)
            self._values = _values

        return None


    def _validate(self, types) -> bool:
        """
        Confirm that labels are either all discrete, or all continuous
        """
        discrete_pattern = r"category|int|str"
        check_discrete = [bool(re.search(discrete_pattern, s)) for s in types]

        if all(check_discrete):
            self._labeltype='discrete'

            return True
        elif not any(check_discrete):
            self._labeltype='continuous'
            
            return True
        else:
            raise ValueError("Labels should be discrete or continous, not both.")

    @staticmethod
    def one_hot_encode(values):
        return pd.get_dummies(values)
    
    @staticmethod
    def collapse(values, sep="_"):
        return np.array(["_".join(row) for row in values])


    
    