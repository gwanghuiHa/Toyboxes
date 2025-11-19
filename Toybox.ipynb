#%%
# Beamline Toybox - Spyder + Colab compatible prototype
#
# Requirements:
#   numpy, matplotlib, ipywidgets
# In Colab:
#   1) Run:  !pip install ipywidgets
#   2) Then: %matplotlib widget   (or %matplotlib notebook)
#
# You can comment out the ipywidgets / display parts when testing in Spyder.

import numpy as np
import matplotlib.pyplot as plt

try:
    from ipywidgets import (
        VBox, HBox, FloatSlider, FloatLogSlider, FloatText, Dropdown,
        Button, RadioButtons, Layout, interactive_output
    )
    from IPython.display import display
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False
    print("ipywidgets not available - interactive UI will not run, but physics functions are usable.")

# ----------------------------
# Physics model / helpers
# ----------------------------

def drift_matrix(L):
    """Linear drift matrix for one plane."""
    return np.array([[1.0, L],
                     [0.0, 1.0]])

def quad_thin_matrix(K):
    """
    Thin-lens quadrupole matrix for one plane.
    K is focusing strength [1/m].
    For x-plane, use +K; for y-plane, use -K.
    """
    return np.array([[1.0, 0.0],
                     [-K, 1.0]])

def build_lattice(quads, z_max, Lq=0.1):
    """
    Build a simple 1D lattice from 0 to z_max using thin-lens quads.
    quads: list of dicts with keys {'z', 'K'} where z is center position.
    Lq is the (effective) physical quad length used for drawing only.
    Returns:
        segments: list of ('drift', L) or ('quad', K) in order
        z_positions: array of z positions for envelope sampling
        element_z: list of z positions where quad lenses are located
    """
    # Sort quads by z
    quads_sorted = sorted(quads, key=lambda q: q['z'])
    element_z = [q['z'] for q in quads_sorted]

    # Segments: drift from 0 to first quad, then quad, etc.
    segments = []
    z_prev = 0.0
    for q in quads_sorted:
        if q['z'] > z_prev:
            segments.append(('drift', q['z'] - z_prev))
            z_prev = q['z']
        # thin lens at q['z']
        segments.append(('quad', q['K']))
    if z_prev < z_max:
        segments.append(('drift', z_max - z_prev))

    # For envelope sampling, use a fixed grid
    z_positions = np.linspace(0.0, z_max, 400)
    return segments, z_positions, element_z, quads_sorted

def propagate_sigma_plane(segments, z_grid, Sigma0, signK=+1):
    """
    Propagate 2x2 sigma matrix through lattice for one plane.
    signK = +1 for x-plane, -1 for y-plane (quad focusing sign).
    Returns:
        sigma(z) = sqrt(<x^2>) along z_grid
    """
    # Current position and matrix
    z_current = 0.0
    M = np.eye(2)
    Sigma_list = []

    # For efficient stepping, walk through segments and z_grid together
    seg_index = 0
    seg_type, seg_val = segments[seg_index]
    remaining = seg_val

    for z in z_grid:
        # Advance from z_current to z
        dz = z - z_current
        while dz > 1e-12:
            if seg_type == 'drift':
                step = min(dz, remaining)
                Md = drift_matrix(step)
                M = Md @ M
                dz -= step
                z_current += step
                remaining -= step
            elif seg_type == 'quad':
                # Apply quad as thin lens at its z-position
                # Only applied once when dz crosses 0 for that seg
                Mq = quad_thin_matrix(signK * seg_val)
                M = Mq @ M
                # Quad has zero length in this model, so we "consume" it
                remaining = 0.0
                dz = dz  # unchanged
            if remaining <= 1e-12:
                # Move to next segment if exists
                seg_index += 1
                if seg_index < len(segments):
                    seg_type, seg_val = segments[seg_index]
                    remaining = seg_val
                else:
                    # No more segments; propagate drift only
                    seg_type, seg_val = 'drift', 1e9
                    remaining = 1e9

        Sigma = M @ Sigma0 @ M.T
        Sigma_list.append(Sigma)

    Sigma_array = np.stack(Sigma_list, axis=0)
    sigma_pos = np.sqrt(Sigma_array[:, 0, 0])  # sqrt(<x^2>)
    return sigma_pos, Sigma_array

def sigma_from_particles(x, xp):
    """Compute 2x2 covariance matrix from particle arrays."""
    X = np.vstack((x, xp))
    cov = np.cov(X)
    return cov

def generate_initial_distribution(N, params):
    """
    Generate initial (x, x', y, y') arrays based on UI params.
    params dict:
        'dist_type': 'Gaussian' or 'Uniform circular'
        'sigx', 'sigy': position rms [m]
        'sigxp', 'sigyp': uncorrelated angle rms [rad]
        'slope_x', 'slope_y': correlation slopes [1/m]
    """
    sigx = params['sigx']
    sigy = params['sigy']
    sigxp = params['sigxp']
    sigyp = params['sigyp']
    slope_x = params['slope_x']
    slope_y = params['slope_y']
    dist_type = params['dist_type']

    if dist_type == 'Gaussian':
        x = np.random.normal(0.0, sigx, N)
        y = np.random.normal(0.0, sigy, N)
    else:  # Uniform circular in x-y
        # Radius ~ 2*sigma as a rough mapping
        R = 2.0 * max(sigx, sigy)
        r = R * np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

    # Uncorrelated angular spread
    xp_unc = np.random.normal(0.0, sigxp, N)
    yp_unc = np.random.normal(0.0, sigyp, N)

    # Add slope correlation: x' = slope_x * x + xp_unc, etc.
    xp = slope_x * x + xp_unc
    yp = slope_y * y + yp_unc

    return x, xp, y, yp

def propagate_particles_to_z(x0, xp0, y0, yp0, segments, z_target):
    """
    Propagate particles to a given z_target using same linear model.
    We only need the total matrix from 0 to z_target.
    """
    # Build matrix Mx, My up to z_target
    z_current = 0.0
    Mx = np.eye(2)
    My = np.eye(2)

    seg_index = 0
    seg_type, seg_val = segments[seg_index]
    remaining = seg_val
    z_target = float(z_target)

    while z_current < z_target - 1e-12:
        dz = min(z_target - z_current, remaining)
        if seg_type == 'drift':
            Md = drift_matrix(dz)
            Mx = Md @ Mx
            My = Md @ My
            z_current += dz
            remaining -= dz
        elif seg_type == 'quad':
            # Thin lens at its z
            Mq_x = quad_thin_matrix(+seg_val)
            Mq_y = quad_thin_matrix(-seg_val)
            Mx = Mq_x @ Mx
            My = Mq_y @ My
            # Quad has no length; mark consumed
            remaining = 0.0

        if remaining <= 1e-12:
            seg_index += 1
            if seg_index < len(segments):
                seg_type, seg_val = segments[seg_index]
                remaining = seg_val
            else:
                # Past last element: pure drift
                seg_type, seg_val = 'drift', 1e9
                remaining = 1e9

    # Apply matrix to particles
    X0 = np.vstack((x0, xp0))
    Y0 = np.vstack((y0, yp0))
    Xz = Mx @ X0
    Yz = My @ Y0
    xz, xpz = Xz[0, :], Xz[1, :]
    yz, ypz = Yz[0, :], Yz[1, :]
    return xz, xpz, yz, ypz

# ----------------------------
# Visualization setup
# ----------------------------

def create_initial_distribution_figure():
    fig, axs = plt.subplots(2, 2, figsize=(7, 6))
    ax_xxprime = axs[0, 1]
    ax_yyprime = axs[1, 0]
    ax_xy = axs[1, 1]
    axs[0, 0].axis('off')
    ax_xxprime.set_xlabel("x [m]")
    ax_xxprime.set_ylabel("x' [rad]")
    ax_yyprime.set_xlabel("y [m]")
    ax_yyprime.set_ylabel("y' [rad]")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    fig.tight_layout()
    return fig, axs

def update_initial_distribution_plots(axs, x, xp, y, yp):
    ax_xxprime = axs[0, 1]
    ax_yyprime = axs[1, 0]
    ax_xy = axs[1, 1]

    for ax in [ax_xxprime, ax_yyprime, ax_xy]:
        ax.cla()

    ax_xxprime.scatter(x, xp, s=2, alpha=0.4)
    ax_xxprime.set_xlabel("x [m]")
    ax_xxprime.set_ylabel("x' [rad]")

    ax_yyprime.scatter(y, yp, s=2, alpha=0.4)
    ax_yyprime.set_xlabel("y [m]")
    ax_yyprime.set_ylabel("y' [rad]")

    ax_xy.scatter(x, y, s=2, alpha=0.4)
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")

    for ax in [ax_xxprime, ax_yyprime, ax_xy]:
        ax.grid(True)

def create_beamline_figure():
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(8, 5))
    gs = GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[2, 1], figure=fig)

    ax_layout = fig.add_subplot(gs[0, :])
    ax_envelope = fig.add_subplot(gs[1, 0])
    ax_cross = fig.add_subplot(gs[1, 1])

    ax_layout.set_ylabel("Element")
    ax_layout.set_xticklabels([])
    ax_envelope.set_xlabel("z [m]")
    ax_envelope.set_ylabel("Beam size [m]")
    ax_cross.set_xlabel("x [m]")
    ax_cross.set_ylabel("y [m]")

    fig.tight_layout()
    return fig, ax_layout, ax_envelope, ax_cross

def draw_beamline(ax_layout, quads, z_max, Lq=0.1, selected_index=None):
    ax_layout.cla()
    ax_layout.set_xlim(0, z_max)
    ax_layout.set_ylim(0, 1)
    ax_layout.set_yticks([])
    ax_layout.set_ylabel("Quads")

    for i, q in enumerate(quads):
        zc = q['z']
        width = Lq
        left = zc - width/2
        color = 'tab:blue'
        if selected_index is not None and i == selected_index:
            color = 'tab:orange'
        ax_layout.add_patch(
            plt.Rectangle((left, 0.3), width, 0.4, color=color, alpha=0.8)
        )
        ax_layout.text(zc, 0.75, f"Q{i}", ha='center', va='bottom', fontsize=8)

    ax_layout.set_xlim(0, z_max)
    ax_layout.grid(True, axis='x', linestyle=':')

def draw_envelope(ax_envelope, z_grid, sigx_z, sigy_z, z_view=None, yscale_factor=1.0):
    ax_envelope.cla()
    ax_envelope.plot(z_grid, sigx_z, label='σx')
    ax_envelope.plot(z_grid, sigy_z, label='σy')
    if z_view is not None:
        ax_envelope.axvline(z_view, color='red', linestyle='--', alpha=0.7)
    ax_envelope.set_xlabel("z [m]")
    ax_envelope.set_ylabel("Beam size [m]")
    # Apply y-scale factor (just multiplies visual range)
    current_max = max(np.max(sigx_z), np.max(sigy_z)) * 1.1
    ax_envelope.set_ylim(0, current_max * yscale_factor)
    ax_envelope.legend()
    ax_envelope.grid(True)

def draw_cross_section(ax_cross, xz, yz):
    ax_cross.cla()
    ax_cross.scatter(xz, yz, s=3, alpha=0.4)
    ax_cross.set_xlabel("x [m]")
    ax_cross.set_ylabel("y [m]")
    if len(xz) > 0:
        max_range = 1.1 * max(np.std(xz), np.std(yz)) * 4
        if max_range <= 0:
            max_range = 1e-3
        ax_cross.set_xlim(-max_range, max_range)
        ax_cross.set_ylim(-max_range, max_range)
    ax_cross.grid(True)

# ----------------------------
# Interactive UI (notebook)
# ----------------------------

def launch_toybox():
    if not IN_NOTEBOOK:
        print("Interactive UI requires ipywidgets and a notebook environment.")
        return

    # ----- initial objects -----
    N_particles = 2000

    # Default initial distribution parameters
    init_params = {
        'dist_type': 'Gaussian',
        'sigx': 1e-3,
        'sigy': 1e-3,
        'sigxp': 1e-4,
        'sigyp': 1e-4,
        'slope_x': 0.0,
        'slope_y': 0.0,
    }

    # Default beamline: one quad at 0.5 m, K = 1.0 [1/m]
    # (You can later rescale to real T/m if you include energy)
    quads = [{'z': 0.5, 'K': 1.0}]
    Lq = 0.1
    z_max_default = 10.0

    # Create figures
    fig_init, axs_init = create_initial_distribution_figure()
    fig_beam, ax_layout, ax_envelope, ax_cross = create_beamline_figure()

    # ----- Widgets -----

    # Initial distribution controls
    dist_type_widget = RadioButtons(
        options=['Gaussian', 'Uniform circular'],
        value='Gaussian',
        description='XY shape:',
        layout=Layout(width='200px')
    )

    sigx_widget = FloatLogSlider(
        value=1e-3, base=10, min=-4, max=-2, step=0.01,
        description='σx [m]:', continuous_update=False
    )
    sigy_widget = FloatLogSlider(
        value=1e-3, base=10, min=-4, max=-2, step=0.01,
        description='σy [m]:', continuous_update=False
    )
    sigxp_widget = FloatLogSlider(
        value=1e-4, base=10, min=-5, max=-2, step=0.01,
        description='σx\' [rad]:', continuous_update=False
    )
    sigyp_widget = FloatLogSlider(
        value=1e-4, base=10, min=-5, max=-2, step=0.01,
        description='σy\' [rad]:', continuous_update=False
    )
    slope_x_widget = FloatSlider(
        value=0.0, min=-10.0, max=10.0, step=0.1,
        description='slope x [1/m]:', continuous_update=False
    )
    slope_y_widget = FloatSlider(
        value=0.0, min=-10.0, max=10.0, step=0.1,
        description='slope y [1/m]:', continuous_update=False
    )

    # Beamline / quad controls
    zmax_widget = FloatText(
        value=z_max_default,
        description='z_max [m]:',
        layout=Layout(width='150px')
    )

    add_quad_button = Button(
        description='Add Quad',
        button_style='success',
        layout=Layout(width='100px')
    )
    remove_quad_button = Button(
        description='Remove Quad',
        button_style='danger',
        layout=Layout(width='110px')
    )

    # Dropdown for selecting quad
    def quad_options():
        return [f"Q{i}: z={q['z']:.2f} m" for i, q in enumerate(quads)]

    quad_select_widget = Dropdown(
        options=quad_options(),
        value=quad_options()[0],
        description='Select quad:',
        layout=Layout(width='200px')
    )

    quad_strength_widget = FloatSlider(
        value=1.0, min=-5.0, max=5.0, step=0.1,
        description='K [1/m]:', continuous_update=False
    )

    quad_z_widget = FloatSlider(
        value=0.5, min=0.1, max=z_max_default - 0.1, step=0.01,
        description='z [m]:', continuous_update=False
    )

    # Envelope / cross-section controls
    z_view_widget = FloatSlider(
        value=1.0, min=0.0, max=z_max_default, step=0.01,
        description='view z [m]:', continuous_update=False
    )

    yscale_widget = FloatSlider(
        value=1.0, min=0.5, max=10.0, step=0.1,
        description='Y-scale:', continuous_update=False
    )

    # ----- State -----
    # We'll refresh these inside a function
    x0 = xp0 = y0 = yp0 = None
    z_grid = None
    sigx_z = sigy_z = None

    # Helpers to access selected quad index
    def get_selected_quad_index():
        label = quad_select_widget.value
        idx = int(label.split(':')[0][1:])
        return idx

    def clamp_quad_positions(index_moved=None):
        """Ensure quads do not overlap; adjust if needed."""
        # Sort by z but keep original order for editing
        quads_sorted = sorted(enumerate(quads), key=lambda iq: iq[1]['z'])
        min_spacing = 0.15  # m, minimum distance between centers
        # Sweep and enforce spacing
        for i in range(1, len(quads_sorted)):
            idx_prev, q_prev = quads_sorted[i-1]
            idx_curr, q_curr = quads_sorted[i]
            if q_curr['z'] - q_prev['z'] < min_spacing:
                q_curr['z'] = q_prev['z'] + min_spacing
        # Also enforce 0+margin and z_max-margin
        zmax = float(zmax_widget.value)
        for idx, q in enumerate(quads):
            q['z'] = min(max(q['z'], 0.1), zmax - 0.1)

    # ----- Core update routine -----

    def recompute_and_redraw(_=None):
        nonlocal x0, xp0, y0, yp0, z_grid, sigx_z, sigy_z

        # Read initial distribution parameters
        init_params['dist_type'] = dist_type_widget.value
        init_params['sigx'] = sigx_widget.value
        init_params['sigy'] = sigy_widget.value
        init_params['sigxp'] = sigxp_widget.value
        init_params['sigyp'] = sigyp_widget.value
        init_params['slope_x'] = slope_x_widget.value
        init_params['slope_y'] = slope_y_widget.value

        # Generate new initial particles
        x0, xp0, y0, yp0 = generate_initial_distribution(N_particles, init_params)
        update_initial_distribution_plots(axs_init, x0, xp0, y0, yp0)
        fig_init.canvas.draw_idle()

        # Lattice and envelope
        zmax = float(zmax_widget.value)
        segments, z_grid, element_z, quads_sorted = build_lattice(quads, zmax, Lq=Lq)

        # Compute sigma0 from particles
        Sigma0_x = sigma_from_particles(x0, xp0)
        Sigma0_y = sigma_from_particles(y0, yp0)

        sigx_z, _ = propagate_sigma_plane(segments, z_grid, Sigma0_x, signK=+1)
        sigy_z, _ = propagate_sigma_plane(segments, z_grid, Sigma0_y, signK=-1)

        # Selected quad index
        sel_idx = get_selected_quad_index()

        # Beamline layout
        draw_beamline(ax_layout, quads, zmax, Lq=Lq, selected_index=sel_idx)

        # Envelope
        z_view = z_view_widget.value
        yscale_factor = yscale_widget.value
        draw_envelope(ax_envelope, z_grid, sigx_z, sigy_z, z_view=z_view,
                      yscale_factor=yscale_factor)

        # Cross-section at z_view
        xz, xpz, yz, ypz = propagate_particles_to_z(x0, xp0, y0, yp0, segments, z_view)
        draw_cross_section(ax_cross, xz, yz)

        fig_beam.canvas.draw_idle()

    # ----- Widget callbacks -----

    def on_add_quad_clicked(b):
        # Add new quad near middle or at last + 0.5 m
        zmax = float(zmax_widget.value)
        if quads:
            z_new = min(quads[-1]['z'] + 0.5, zmax - 0.5)
        else:
            z_new = min(0.5, zmax - 0.5)
        quads.append({'z': z_new, 'K': 1.0})
        clamp_quad_positions()
        # Update dropdown
        quad_select_widget.options = quad_options()
        quad_select_widget.value = quad_options()[-1]
        on_quad_selection_change(None)

    def on_remove_quad_clicked(b):
        if not quads:
            return
        idx = get_selected_quad_index()
        if len(quads) == 1:
            # Keep at least one quad to avoid degenerate UI
            return
        quads.pop(idx)
        # Update dropdown
        quad_select_widget.options = quad_options()
        quad_select_widget.value = quad_options()[0]
        on_quad_selection_change(None)

    def on_quad_selection_change(change):
        # Update strength and z sliders to match selected quad
        idx = get_selected_quad_index()
        quad_strength_widget.value = quads[idx]['K']
        zmax = float(zmax_widget.value)
        quad_z_widget.max = max(0.2, zmax - 0.1)
        quad_z_widget.value = quads[idx]['z']
        recompute_and_redraw()

    def on_quad_strength_changed(change):
        idx = get_selected_quad_index()
        quads[idx]['K'] = quad_strength_widget.value
        recompute_and_redraw()

    def on_quad_z_changed(change):
        idx = get_selected_quad_index()
        quads[idx]['z'] = quad_z_widget.value
        clamp_quad_positions(index_moved=idx)
        # Refresh z slider limits and dropdown text
        quad_select_widget.options = quad_options()
        recompute_and_redraw()

    def on_zmax_changed(change):
        # Update z_view max and quad position limits
        zmax = float(zmax_widget.value)
        if zmax <= 0.5:
            zmax_widget.value = 1.0
            zmax = 1.0
        z_view_widget.max = zmax
        quad_z_widget.max = max(0.2, zmax - 0.1)
        clamp_quad_positions()
        quad_select_widget.options = quad_options()
        recompute_and_redraw()

    def on_z_view_changed(change):
        recompute_and_redraw()

    def on_yscale_changed(change):
        recompute_and_redraw()

    # Connect callbacks
    add_quad_button.on_click(on_add_quad_clicked)
    remove_quad_button.on_click(on_remove_quad_clicked)
    quad_select_widget.observe(on_quad_selection_change, names='value')
    quad_strength_widget.observe(on_quad_strength_changed, names='value')
    quad_z_widget.observe(on_quad_z_changed, names='value')
    zmax_widget.observe(on_zmax_changed, names='value')
    z_view_widget.observe(on_z_view_changed, names='value')
    yscale_widget.observe(on_yscale_changed, names='value')

    # Also initial distribution widgets
    for w in [dist_type_widget, sigx_widget, sigy_widget,
              sigxp_widget, sigyp_widget, slope_x_widget, slope_y_widget]:
        w.observe(lambda change: recompute_and_redraw(), names='value')

    # Layout
    left_column = VBox([
        dist_type_widget,
        sigx_widget,
        sigy_widget,
        sigxp_widget,
        sigyp_widget,
        slope_x_widget,
        slope_y_widget,
    ])

    quad_controls = VBox([
        HBox([add_quad_button, remove_quad_button]),
        quad_select_widget,
        quad_strength_widget,
        quad_z_widget,
        zmax_widget,
        z_view_widget,
        yscale_widget,
    ])

    ui = HBox([left_column, quad_controls])

    display(ui)

    # Initial draw
    recompute_and_redraw()

# If you’re in a notebook, call this:
# launch_toybox()
