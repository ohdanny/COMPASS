import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import pandas as pd

def spot_image(
    values,
    grid_n: int,
    rows: int,
    cols: int,
    main: str,
    value_label: str,
    log: logging.Logger,
    ax_titles: list[str] | None = None,
    zlim: tuple[float, float] = (0.0, 1.0),
    palette="viridis",
    save_path=None,
):
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 4 * rows),  # Adjusted for better square proportions
        constrained_layout=True,
    )

    norm = Normalize(vmin=zlim[0], vmax=zlim[1])
    axes = np.atleast_1d(axes).flatten()  # type: ignore

    num_plots = values.shape[1]
    im = None  # We will save the last image drawn to build the shared colorbar

    for k in range(num_plots):
        ax = axes[k]
        grid_data = values[:, k].reshape((grid_n, grid_n))

        im = ax.imshow(
            grid_data.T, cmap=palette, norm=norm, origin="lower", extent=[0, 1, 0, 1]
        )

        # 1. Use ax_titles if provided, else use default naming
        if ax_titles is not None and k < len(ax_titles):
            ax.set_title(ax_titles[k], fontsize=10, fontweight="bold")
        else:
            title_parts = [main, str(k + 1)]
            ax.set_title(" ".join(filter(None, title_parts)), fontsize=10)

        # 2. Add y-axis label like the R plot (leaves the ticks visible)
        ax.set_ylabel("v")
        # ax.set_xlabel("u") # Uncomment if you want 'u' on the x-axis too

    # 3. Add a SINGLE shared colorbar outside the loop
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=15)
        if value_label:
            # Puts the label at the top of the colorbar like in R
            cbar.ax.set_title(value_label, pad=12, fontsize=16, fontweight="bold")

    for k in range(num_plots, len(axes)):
        axes[k].axis("off")

    plt.suptitle(main, fontsize=22, fontweight="bold")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info(f"Saved plot to {save_path}")

    plt.close()


def plot_cell_type(
        values, 
        grid_n, 
        rows, 
        cols, 
        log,
        out,
):

    spot_image(
        values=values, 
        grid_n=grid_n, 
        rows=rows, 
        cols=cols, 
        log=log,
        main="Cell Type",
        ax_titles=[
            "cell type 1 (Gaussian bump near (0.25,0.25))",
            "cell type 2 (Gaussian bump near (0.75,0.75))",
            "cell type 3 (reference, baseline)",
        ],
        value_label="$W_{st}$",
        save_path=out)

def save_csv(rows, path, log):
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info(f"Wrote {path}")



def plot_scenarios_grid(values, grid_n, ax_titles, value_label, main, out, log):
    """Thin wrapper around spot_image for the 3x3 scenario grids."""
    spot_image(
        values=values,
        grid_n=grid_n,
        rows=3,
        cols=3,
        ax_titles=ax_titles,
        value_label=value_label,
        main=main,
        log=log,
        save_path=out,
    )
    


def plot_null_qq(p_null, out, log):
    p_null = np.sort(p_null[np.isfinite(p_null)])
    if not len(p_null):
        log.info("no finite null p-values; skipping QQ")
        return
    n = len(p_null)
    exp_p = (np.arange(1, n + 1) - 0.5) / n
    x, y = -np.log10(exp_p), -np.log10(p_null)
    hi = max(x.max(), y.max()) + 0.2
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, c="grey", s=18)
    plt.plot([0, hi], [0, hi], "r--", lw=2)
    plt.xlim(0, hi)
    plt.ylim(0, hi)
    plt.gca().set_aspect("equal")          # carries the QQ-square fix forward
    plt.xlabel("Expected -log10(p)") 
    plt.ylabel("Observed -log10(p)")
    plt.title("Null scenario QQ (omnibus)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"Wrote {out}")

def plot_power_bar(power_rows, out, log):
    labels = [f"{r['scenario']}\n{r['cell_type']}" for r in power_rows]
    vals = [r["power"] for r in power_rows]
    colors = ["#3B7EA1" if r["active_truth"] else "grey" for r in power_rows]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, vals, color=colors)
    plt.axhline(0.05, color="red", ls="--")
    plt.ylim(0, 1)
    plt.ylabel("Pr(p < 0.05)")
    plt.title("Cell-type-specific interaction power")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"Wrote {out}")



def plot_eta_decomposition(comps, grid_n, scenario, out, log):
    Km = comps["eta"].shape[1]
    col_order = ["mix", "sh", "int1", "int2", "eta"]
    titles = [
        r"mixture $\alpha+X\beta$",
        r"shared $B\xi_0$",
        r"t=1 $w_1 B\xi_1$",
        r"t=2 $w_2 B\xi_2$",
        r"total $\eta$",
    ]
    zmax = float(np.max(np.abs(np.concatenate([comps[c].ravel() for c in col_order]))))
    fig, axes = plt.subplots(Km, 5, figsize=(20, 4 * Km), constrained_layout=True)
    axes = np.atleast_2d(axes)
    im = None
    for ki in range(Km):
        for cj, name in enumerate(col_order):
            ax = axes[ki, cj]
            im = ax.imshow(
                comps[name][:, ki].reshape(grid_n, grid_n).T,
                cmap="RdBu_r",
                vmin=-zmax,
                vmax=zmax,
                origin="lower",
                extent=[0, 1, 0, 1],
            )
            if ki == 0:
                ax.set_title(titles[cj], fontsize=11)
            if cj == 0:
                ax.set_ylabel(f"isoform k={ki+1}", fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=0.7) # type: ignore
    fig.suptitle(
        f"Eta decomposition — {scenario}  (cols 1–4 sum to col 5)",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Wrote {out}")