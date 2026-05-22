import jax.numpy as jnp
from patsy import dmatrix #type: ignore


def build_bspline_basis(
coords:jnp.ndarray,
    df_per_axis:int = 2,
    orthonormalize:bool = True,
    unit_variance:bool = True
    ):
    u = coords[:,0]
    v = coords[:,1]
    if df_per_axis < 3:
        Bu = dmatrix(
            "bs(u, df={0}, degree={0}, include_intercept=False) - 1".format(
                df_per_axis
            ),
            {"u": u},
        )
        Bv = dmatrix(
            "bs(v, df={0}, degree={0}, include_intercept=False) - 1".format(
                df_per_axis
            ),
            {"v": v},
        )
    else:
        Bu = dmatrix(
        "bs(u, df={0}, include_intercept=False) - 1".format(
            df_per_axis
        ),
        {"u":u}
        )

        Bv = dmatrix(
        "bs(v, df={0}, include_intercept=False) - 1".format(
            df_per_axis
        ),
        {"v":v}
        )

    B_raw = (Bu[:, :, None] * Bv[:, None, :]).reshape(coords.shape[0], -1)
    B_c = B_raw - B_raw.mean(axis=0)
    if orthonormalize:
        # Orthonormalize the basis functions using QR decomposition
        Q, R = jnp.linalg.qr(B_c)
        rank = jnp.linalg.matrix_rank(R)
        B_out = Q[:, :rank]
        if unit_variance:
            # Scale the orthonormalized basis functions to have unit variance
            B_out = B_out / jnp.sqrt(jnp.var(B_out, axis=0))
    else:
        B_out = B_c
    return B_out

def build_basis_to_rank(coords:jnp.ndarray,
                        r_target:int):
    df_axis = max(2,int(jnp.ceil(jnp.sqrt(r_target))))
    while True:
        B = build_bspline_basis(coords,df_per_axis=df_axis,orthonormalize=True)
        if B.shape[1] >= r_target or df_axis > 6:
            break
        df_axis+=1
        
    return B[:,:min(r_target,B.shape[1])]

# build_bspline_basis(coords=jnp.array([[0.1,0.2],[0.3,0.4],[0.5,0.6]]), df_per_axis=3)

# i will do it myself l8r to learn ab it
# def build_bspline_basis(
        # coords:jnp.ndarray,
    #     df_per_axis:int = 2,
    #     orthonormalize:bool = True,
    #     unit_variance:bool = True
    #     ):
    
    # if coords.shape[1] != 2:
    #     raise ValueError("Coordinates must be x,y (2 columns).")
    # u = coords[:,1]
    # v = coords[:,2]
