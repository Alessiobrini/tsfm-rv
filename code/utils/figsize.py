import matplotlib.pyplot as plt

def set_size(width, fraction=1, subplots=(1, 1)) -> tuple:
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    # fig_height_in = fig_width_in * golden_ratio
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
    
def create_figure(width=800):

    return plt.subplots(figsize=set_size(width=width))


def create_figures(nrows, ncols, width=800, tupsize=None):
    if width:
        return plt.subplots(nrows=nrows, ncols=ncols, figsize=set_size(width=width))
    else:
        return plt.subplots(nrows=nrows, ncols=ncols, figsize=tupsize)
        
        
def create_sharedy_figure(width=800,figsize=None):
    if figsize:
        fig,ax1 = plt.subplots(figsize=figsize)
    else:
        fig,ax1 = plt.subplots(figsize=set_size(width=width))
    
    ax2 = ax1.twinx() 
    
    return fig,ax1,ax2