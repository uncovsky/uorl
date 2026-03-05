import matplotlib.pyplot as plt

"""
TeX setup
"""
latex_paperwidth = 597.50787   # A4 paper width in TeX points (210 mm)
latex_paperheight = 845.04684  # A4 paper height in TeX points (297 mm)
latex_textwidth = 361.34999    # Example value from your setup

tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

plt.rcParams.update(tex_fonts)

def set_size(width_fraction=1.0, height_fraction=None, use_textwidth_for_width=True, subplots=(1, 1)):
    # Determine base width in points
    if use_textwidth_for_width:
        base_width_pt = latex_textwidth
    else:
        base_width_pt = latex_paperwidth

    fig_width_pt = base_width_pt * width_fraction
    inches_per_pt = 1 / 72.27  # PostScript point to inch conversion
    fig_width_in = fig_width_pt * inches_per_pt

    if height_fraction is not None:
        # Explicit height as fraction of paper height
        fig_height_pt = latex_paperheight * height_fraction
        fig_height_in = fig_height_pt * inches_per_pt
    else:
        # default to square
        fig_height_in = fig_width_in

    return (fig_width_in, fig_height_in)
