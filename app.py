import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

st.set_page_config(
    page_title="Simulasi Bola Elektromagnetik & Magnus",
    layout="wide",
)

st.title("Simulasi Bola Berputar dalam Medan Elektromagnetik & Efek Magnus")

# --- Sidebar: Parameter Fisik ---
st.sidebar.header("Parameter Fisik")
m     = st.sidebar.number_input("Massa m (kg)", value=1.0, step=0.1)
q     = st.sidebar.number_input("Muatan q (C)", value=0.0, step=0.1)
R     = st.sidebar.number_input("Jari-jari R (m)", value=0.2, step=0.05)
rho   = st.sidebar.number_input("Densitas udara ρ_air (kg/m³)", value=1.2, step=0.1)
Cm    = st.sidebar.number_input("Koefisien Magnus C_m", value=1.0, step=0.1)
g_vec = np.array([0., 0., -9.8])

st.sidebar.header("Medan Elektromagnetik")
B0 = st.sidebar.slider("B₀ (T)", 0.0, 2.0, 0.5, 0.1)
k  = st.sidebar.slider("Gradien k (T/m)", 0.0, 1.0, 0.1, 0.05)
Ex = st.sidebar.number_input("Eₓ (V/m)", value=0.1, step=0.1)
Ey = st.sidebar.number_input("E_y (V/m)", value=0.0, step=0.1)
Ez = st.sidebar.number_input("E_z (V/m)", value=0.0, step=0.1)
E  = np.array([Ex, Ey, Ez])

st.sidebar.header("Kondisi Awal")
x0, y0, z0 = st.sidebar.columns(3)
with x0: X0 = st.number_input("x₀ (m)", value=0.0)
with y0: Y0 = st.number_input("y₀ (m)", value=0.0)
with z0: Z0 = st.number_input("z₀ (m)", value=0.0)

vx0, vy0, vz0 = st.sidebar.columns(3)
with vx0: VX0 = st.number_input("vₓ₀ (m/s)", value=5.0)
with vy0: VY0 = st.number_input("v_y₀ (m/s)", value=0.0)
with vz0: VZ0 = st.number_input("v_z₀ (m/s)", value=10.0)

ox0, oy0, oz0 = st.sidebar.columns(3)
with ox0: OX0 = st.number_input("ωₓ₀ (rad/s)", value=0.0)
with oy0: OY0 = st.number_input("ω_y₀ (rad/s)", value=50.0)
with oz0: OZ0 = st.number_input("ω_z₀ (rad/s)", value=0.0)

st.sidebar.header("Waktu Simulasi")
t_start = 0.0
t_end   = st.sidebar.number_input("Durasi (s)", value=10.0)
n_pts   = st.sidebar.slider("Jumlah Titik", 100, 5000, 1000)

if st.sidebar.button("🔄 Jalankan Simulasi"):
    # --- Definisi Fungsi Fisika ---
    def magnetic_field(pos):
        return np.array([0., 0., B0 + k*pos[2]])
    def magnetic_force(pos, omega):
        m_vec = (q * R**2 / 5) * omega
        return np.array([0., 0., m_vec[2] * k])
    def magnus_force(vel, omega):
        Km = rho * Cm * np.pi * R**3 / 3
        return Km * np.cross(omega, vel)
    def system(t, y):
        pos, vel, omega = y[:3], y[3:6], y[6:9]
        B = magnetic_field(pos)
        F_lorentz = q * (E + np.cross(vel, B))
        F_mag     = magnetic_force(pos, omega)
        F_magnus  = magnus_force(vel, omega)
        F_grav    = m * g_vec
        total = F_lorentz + F_mag + F_magnus + F_grav
        dvdt = total / m
        dwdt = (q/(2*m)) * np.cross(omega, B)
        return np.hstack((vel, dvdt, dwdt))

    # --- Solusi Numerik ---
    t_eval = np.linspace(t_start, t_end, int(n_pts))
    y0 = [X0, Y0, Z0, VX0, VY0, VZ0, OX0, OY0, OZ0]
    sol = solve_ivp(system, (t_start, t_end), y0, t_eval=t_eval, rtol=1e-6)

    # --- Ambil Data ---
    x, y, z       = sol.y[0], sol.y[1], sol.y[2]
    ωx, ωy, ωz    = sol.y[6], sol.y[7], sol.y[8]
    Fm_x, Fm_y, Fm_z = [], [], []
    for i in range(len(t_eval)):
        vel   = sol.y[3:6, i]
        omega = sol.y[6:9, i]
        Fm = magnus_force(vel, omega)
        Fm_x.append(Fm[0]); Fm_y.append(Fm[1]); Fm_z.append(Fm[2])

    # --- Tabs untuk Visualisasi ---
    tab1, tab2, tab3 = st.tabs(["📈 Lintasan 3D", "🔄 ω vs Waktu", "🌪️ Gaya Magnus"])
    with tab1:
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=4, color=z, colorscale='Viridis', opacity=0.8)
        )])
        fig.update_layout(margin=dict(l=0,r=0,b=0,t=30), scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'
        ), title="Lintasan 3D Bola")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.line_chart({'ωx': ωx, 'ωy': ωy, 'ωz': ωz}, height=400)

    with tab3:
        st.line_chart({'Fm_x': Fm_x, 'Fm_y': Fm_y, 'Fm_z': Fm_z}, height=400)

    # --- Ringkasan Hasil ---
    st.markdown("**Posisi Akhir (t = %.1f s):** (%.2f, %.2f, %.2f) m" %
                (t_end, x[-1], y[-1], z[-1]))
    st.markdown("**Kecepatan Sudut Akhir:** (%.2f, %.2f, %.2f) rad/s" %
                (ωx[-1], ωy[-1], ωz[-1]))
