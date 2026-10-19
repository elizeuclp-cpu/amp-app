import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# CONFIGURACAO DA PAGINA
# ============================================================
st.set_page_config(
    page_title="Ampacidade IEEE 738 - Linhas Aéreas",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("⚡ Ampacidade de Condutores - IEEE 738-2006")
st.caption("Cálculo térmico de regime permanente e curva de ampacidade de condutores aéreos.")
st.markdown("---")

# ============================================================
# BANCO DE DADOS DE CABOS (ESPAÇO RESERVADO)
# ============================================================
# D = Diâmetro externo em metros
# R25 = Resistência CA a 25°C em Ohms/metro
# R75 = Resistência CA a 75°C em Ohms/metro

CABOS = {
    "Linnet (336,4 MCM)": {"D": 0.018288, "R25": 0.0001695, "R75": 0.0002028},
    # --- Adicione novos cabos abaixo (Descomente e preencha) ---
    # "Grosbeak (636 MCM)": {"D": 0.025150, "R25": 0.0001080, "R75": 0.0001290},
    # "Raven (1/0 AWG)":    {"D": 0.010110, "R25": 0.0007090, "R75": 0.0008460},
}

# ============================================================
# FUNÇÕES MATEMÁTICAS (IEEE 738)
# ============================================================
MU_F = 1.86e-5       # Viscosidade dinâmica do ar (kg/m s)
RHO_F = 1.16         # Densidade do ar base (kg/m^3)
K_F = 0.026          # Condutividade térmica do ar (W/m K)

def calcular_temperatura(I, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75):
    pressao_nivel_mar = 101325 
    pressao_local = pressao_nivel_mar * (1 - (2.25577e-5 * elevacao))**5.25588
    rho_ar = (pressao_local * 0.003484) / (Ta + 273.15) 
    
    qs_linear = alpha_abs * qs_global * D
    alpha_R = (R_75 - R_25) / (R_25 * 50) 
    
    def balanco_termico(Tc):
        R_ca_tc = R_25 * (1 + alpha_R * (Tc - 25))
        q_joule = (I**2) * R_ca_tc
        
        q_rad = 17.8 * D * epsilon * (((Tc + 273.15)/100)**4 - ((Ta + 273.15)/100)**4)
        
        Re = (D * rho_ar * V) / MU_F
        phi = np.radians(angulo_vento)
        K_angle = 1.194 - np.cos(phi) + 0.194 * np.cos(2*phi) + 0.368 * np.sin(2*phi)
        
        if Re < 1000:
            q_c_forcada = K_angle * (1.01 + 1.35 * (Re**0.52)) * K_F * (Tc - Ta)
        else:
            q_c_forcada = K_angle * 0.754 * (Re**0.6) * K_F * (Tc - Ta)
            
        q_c_natural = 3.645 * (rho_ar**0.5) * (D**0.75) * ((Tc - Ta)**1.25)
        q_conv = max(q_c_forcada, q_c_natural)
        
        erro = (q_joule + qs_linear) - (q_conv + q_rad)
        return erro, q_joule, q_conv, q_rad, qs_linear

    T_min = Ta 
    T_max = 250.0 
    
    if balanco_termico(T_max)[0] > 0:
        return T_max, *balanco_termico(T_max)[1:]

    for _ in range(100):
        T_medio = (T_min + T_max) / 2
        erro, q_j, q_c, q_r, q_s = balanco_termico(T_medio)
        if erro > 0:
            T_min = T_medio
        else:
            T_max = T_medio
        if abs(erro) < 0.001:
            break
            
    _, q_j, q_c, q_r, q_s = balanco_termico(T_medio)
    return T_medio, q_j, q_c, q_r, q_s

def calcular_ampacidade(Tc_alvo, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75):
    pressao_nivel_mar = 101325 
    pressao_local = pressao_nivel_mar * (1 - (2.25577e-5 * elevacao))**5.25588
    rho_ar = (pressao_local * 0.003484) / (Ta + 273.15) 
    
    qs_linear = alpha_abs * qs_global * D
    alpha_R = (R_75 - R_25) / (R_25 * 50) 
    R_ca_tc = R_25 * (1 + alpha_R * (Tc_alvo - 25))
    
    q_rad = 17.8 * D * epsilon * (((Tc_alvo + 273.15)/100)**4 - ((Ta + 273.15)/100)**4)
    
    Re = (D * rho_ar * V) / MU_F
    phi = np.radians(angulo_vento)
    K_angle = 1.194 - np.cos(phi) + 0.194 * np.cos(2*phi) + 0.368 * np.sin(2*phi)
    
    if Re < 1000:
        q_c_forcada = K_angle * (1.01 + 1.35 * (Re**0.52)) * K_F * (Tc_alvo - Ta)
    else:
        q_c_forcada = K_angle * 0.754 * (Re**0.6) * K_F * (Tc_alvo - Ta)
        
    q_c_natural = 3.645 * (rho_ar**0.5) * (D**0.75) * ((Tc_alvo - Ta)**1.25)
    q_conv = max(q_c_forcada, q_c_natural)
    
    q_liquido = q_conv + q_rad - qs_linear
    
    if q_liquido <= 0:
        return 0.0, 0.0, q_conv, q_rad, qs_linear
        
    I_max = np.sqrt(q_liquido / R_ca_tc)
    q_joule = (I_max**2) * R_ca_tc
    return I_max, q_joule, q_conv, q_rad, qs_linear

# ============================================================
# INTERFACE
# ============================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("⚙️ 1. CONFIGURAÇÕES GERAIS E CLIMA")
    
    modo_calculo = st.radio(
        "O que você deseja calcular?",
        ["Temperatura (dada a Corrente)", "Ampacidade (dada a Temperatura)"],
        horizontal=True
    )
    
    if modo_calculo == "Temperatura (dada a Corrente)":
        corrente = st.number_input("Corrente CA (A)", min_value=0.0, value=450.0, step=10.0)
        temp_alvo = 90.0
    else:
        temp_alvo = st.number_input("Temperatura Alvo (°C)", min_value=10.0, max_value=150.0, value=90.0, step=5.0)
        corrente = 0.0

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        temp_ar = st.number_input("Temp. do Ar (°C)", value=28.0, step=1.0)
        vel_vento = st.number_input("Vel. do Vento (m/s)", min_value=0.0, value=0.6, step=0.1)
        elevacao = st.number_input("Elevação (m)", value=600.0, step=50.0)
    with c2:
        ang_vento = st.number_input("Ângulo do Vento (°)", min_value=0.0, max_value=90.0, value=90.0, step=10.0)
        rad_solar = st.number_input("Radiação Solar (W/m²)", min_value=0.0, value=1200.0, step=100.0)
        
with col2:
    st.subheader("🔌 2. PROPRIEDADES DO CONDUTOR")
    
    cabo_nome = st.selectbox("Selecione o Cabo", list(CABOS.keys()))
    cabo_dados = CABOS[cabo_nome]
    
    diametro = st.number_input("Diâmetro Externo D (m)", min_value=0.001, value=cabo_dados["D"], format="%.6f")
    r25 = st.number_input("R CA a 25°C (Ω/m)", min_value=0.000001, value=cabo_dados["R25"], format="%.7f")
    r75 = st.number_input("R CA a 75°C (Ω/m)", min_value=0.000001, value=cabo_dados["R75"], format="%.7f")
    
    st.markdown("---")
    st.markdown("**Propriedades de Superfície**")
    c3, c4 = st.columns(2)
    with c3:
        alpha_abs = st.number_input("Absortividade", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    with c4:
        epsilon = st.number_input("Emissividade", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

st.markdown("---")

# ============================================================
# BOTÕES E CONTROLE DE ESTADO
# ============================================================
if "calculo_liberado" not in st.session_state:
    st.session_state["calculo_liberado"] = False

_hash_atual = hash((modo_calculo, corrente, temp_alvo, temp_ar, vel_vento, ang_vento, rad_solar, elevacao, cabo_nome, diametro, r25, r75, alpha_abs, epsilon))
if st.session_state.get("hash_calc") != _hash_atual:
    st.session_state["calculo_liberado"] = False
    st.session_state["hash_calc"] = _hash_atual

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    testar = st.button("🧭 Validar Parâmetros", use_container_width=True)
    if testar:
        st.session_state["calculo_liberado"] = True

with col_btn2:
    calcular = st.button("🔍 Calcular", type="primary", use_container_width=True, disabled=not st.session_state["calculo_liberado"])

# ============================================================
# PROCESSAMENTO E RESULTADOS
# ============================================================
if st.session_state["calculo_liberado"]:
    
    if testar:
        st.success("Parâmetros validados! Clique em **🔍 Calcular** para processar.")
    
    if calcular:
        st.markdown("---")
        
        if modo_calculo == "Temperatura (dada a Corrente)":
            Tc, q_j, q_c, q_r, q_s = calcular_temperatura(corrente, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75)
            titulo_res = f"Temperatura do Condutor: {Tc:.2f} °C"
            valor_principal = Tc
            unidade_principal = "°C"
            I_plot = corrente
            T_plot = Tc
        else:
            I_max, q_j, q_c, q_r, q_s = calcular_ampacidade(temp_alvo, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75)
            titulo_res = f"Ampacidade Máxima: {I_max:.1f} A"
            valor_principal = I_max
            unidade_principal = "A"
            I_plot = I_max
            T_plot = temp_alvo
            
        st.subheader("🎯 RESULTADO PRINCIPAL")
        st.markdown(f"<h2 style='color: #00CC96;'>{titulo_res}</h2>", unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calor Joule (I²R)", f"{q_j:.2f} W/m")
        m2.metric("Perda Convecção", f"{q_c:.2f} W/m")
        m3.metric("Perda Radiação", f"{q_r:.2f} W/m")
        m4.metric("Ganho Solar", f"{q_s:.2f} W/m")
        
        tab_graf, tab_tabela, tab_memorial = st.tabs(["📈 Gráfico de Ampacidade", "📊 Tabela de Pontos", "ℹ️ Memorial de Cálculo"])
        
        with tab_graf:
            correntes = np.linspace(0, max(I_plot * 1.5, 500), 40)
            temps = []
            for c in correntes:
                t_calc, _, _, _, _ = calcular_temperatura(c, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75)
                temps.append(t_calc)
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=correntes, y=temps, mode='lines', name='Curva do Cabo', line=dict(color='royalblue', width=3)))
            
            if valor_principal <= 100:
                fig.add_trace(go.Scatter(x=[I_plot], y=[T_plot], mode='markers+text', name='Ponto Atual', 
                                         text=[f"{T_plot:.1f} °C / {I_plot:.0f} A"], textposition="top center",
                                         marker=dict(color='red', size=12)))
            
            fig.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="Limite 90°C")
            
            fig.update_layout(
                title="Curva de Ampacidade (IEEE 738-2006)",
                xaxis_title="Corrente CA (A)",
                yaxis_title="Temperatura (°C)",
                yaxis=dict(range=[temp_ar, 110]),
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab_tabela:
            st.write("Amostragem da curva de ampacidade:")
            df_curva = pd.DataFrame({"Corrente (A)": correntes, "Temperatura (°C)": temps})
            st.dataframe(df_curva.style.format(precision=2), use_container_width=True, height=300)
            
        with tab_memorial:
            st.markdown("### Fórmulas do IEEE 738-2006")
            st.latex(r"I = \sqrt{\frac{q_c + q_r - q_s}{R(T_c)}}")
            
            st.markdown("**Densidade do Ar Corrigida:**")
            st.latex(r"\rho = \frac{0.003484 \cdot P_{local}}{T_a + 273.15}")
            st.latex(r"P_{local} = 101325 \left(1 - 2.25577 \times 10^{-5} \cdot h \right)^{5.25588}")
            
            st.markdown("**Número de Reynolds:**")
            st.latex(r"Re = \frac{D \cdot \rho \cdot V}{\mu_f}")
            
            st.markdown("**Convecção Forçada (Baixa Velocidade, Re < 1000):**")
            st.latex(r"q_{c1} = K_{angle} [1.01 + 1.35 Re^{0.52}] k_f (T_c - T_a)")
            
            st.markdown("**Convecção Forçada (Alta Velocidade, Re ≥ 1000):**")
            st.latex(r"q_{c2} = K_{angle} [0.754 Re^{0.6}] k_f (T_c - T_a)")
            
            st.markdown("**Fator de Ângulo do Vento (\( \phi \)):**")
            st.latex(r"K_{angle} = 1.194 - \cos(\phi) + 0.194 \cos(2\phi) + 0.368 \sin(2\phi)")
            
            st.markdown("**Convecção Natural:**")
            st.latex(r"q_{cn} = 3.645 \rho^{0.5} D^{0.75} (T_c - T_a)^{1.25}")
            st.caption("O modelo adota a convecção forçada se ela for maior que a natural, e vice-versa.")
            
            st.markdown("**Radiação:**")
            st.latex(r"q_r = 17.8 \cdot D \cdot \varepsilon \left[ \left(\frac{T_c+273}{100}\right)^4 - \left(\frac{T_a+273}{100}\right)^4 \right]")
            
            st.markdown("**Ganho Solar:**")
            st.latex(r"q_s = \alpha \cdot Q_{se} \cdot D")
