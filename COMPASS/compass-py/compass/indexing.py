from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class ParamIndex:
    """
    Args:
        K (int): number of isoform events
        T (int): number of cell types
        r (int): dimension of spatial basis. Should be very low compared to number of spots in data.
    """
    K:int 
    T:int 
    r:int
    full:bool=True

    def __post_init__(self):
        if self.full:
            self.n_param_per_k = self.T * (1 + self.r) # = 1 + (T-1) + rT
            # alpha, T-1 beta, rT xi 
        else:
            self.n_param_per_k = self.T + self.r # = 1 + (T-1) + r
            # 1 alpha, T-1 beta, r xi 

        self.alpha = []
        self.beta = []
        self.xi0 = []


        # (K-1,)
        offs = jnp.arange(self.K-1) * self.n_param_per_k
        self.alpha_idx = offs

        # (K-1, T-1) isoforms x cell types
        # need +1 because alpha takes the first slot for each isoform
        self.beta_idx = offs[:, None] + 1 + jnp.arange(self.T - 1)

        # (K-1, r) isoforms x spatial basis
        # need +T because alpha and beta take the first T slots for each isoform
        self.xi0_idx = offs[:, None] + self.T + jnp.arange(self.r)

        self.xi_int_idx = None
        if self.full and self.T > 1:
            # (K-1, T-1, r) isoforms x cell types x spatial basis
            self.xi_int_idx = (
            offs[:, None, None] # (K-1, 1, 1)
            + self.T 
            # because alpha and beta take the first T slots for each isoform
            + self.r
            # because xi0 takes the next r slots for each isoform
            + jnp.arange(self.T - 1)[None,:, None] * self.r # (1,T-1, 1)
            + jnp.arange(self.r)[None, None, :] # (1, 1,r)
        )

        self.kappa_idx = (self.K-1)*self.n_param_per_k
        self.total_params = self.kappa_idx + 1
    
    def unpack(self,theta: jnp.ndarray):
        assert len(theta) == self.total_params, f"Expected {self.total_params} parameters but got {len(theta)}"

        # (K-1,)
        alpha = theta[self.alpha_idx]
        # (K-1, T-1)
        beta = theta[self.beta_idx]
        # (K-1, r)
        xi0 = theta[self.xi0_idx]

        kappa = theta[self.kappa_idx]
        xi_int = None
        if self.xi_int_idx is not None:
            # (K-1, T-1, r)
            xi_int = theta[self.xi_int_idx]

        return alpha, beta, xi0, xi_int, kappa

    def gamma(self, theta:jnp.ndarray):
        reg = theta[:self.kappa_idx]
        gammas = reg.reshape((self.K-1, self.n_param_per_k)).T
        return gammas
    
    def xi_block_celltype(self, t: int) -> jnp.ndarray:
        if self.xi_int_idx is None:
            raise ValueError("xi_block_celltype requires full=True with T>1")
        # r(K-1) parameters for the interaction of cell type t with all spatial basis functions and isoforms
        return self.xi_int_idx[:, t, :].reshape(-1)    

    def xi_block_pair(self, t: int, k: int) -> jnp.ndarray:
        if self.xi_int_idx is None:
            raise ValueError("xi_block_pair requires full=True with T>1")
        # r parameters for the interaction of cell type t with all spatial basis functions for isoform k
        return self.xi_int_idx[k, t, :]
            
    def __len__(self):
        return self.total_params
        
    def xi_flat(self) -> jnp.ndarray | None:
        if self.xi_int_idx is not None:
            return self.xi_int_idx.flatten()
        
    def __eq__(self,other):
        return(
            isinstance(other,ParamIndex) and
            (self.K, self.T, self.r, self.full) == (other.K, other.T, other.r, other.full)
        )
    def __hash__(self):
        return hash((self.K, self.T, self.r, self.full))