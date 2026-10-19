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
st.caption("Cálculo térmico de regime permanente e transitório de condutores aéreos.")
st.markdown("---")

# ============================================================
# BANCO DE DADOS DE CABOS (ESPAÇO RESERVADO)
# ============================================================
# D = Diâmetro externo em metros
# R25 = Resistência CA a 25°C em Ohms/metro
# R75 = Resistência CA a 75°C em Ohms/metro
# Cp = Capacidade térmica total em J/(m·°C) [Outer + Core]

CABOS = {
    "Linnet (336,4 MCM)": {
        "D": 0.018288, 
        "R25": 0.0001695, 
        "R75": 0.0002028,
        "Cp": 552.81  # 449.771 (Alumínio) + 103.039 (Aço)
    },
    # --- Adicione novos cabos abaixo (Descomente e preencha) ---
    # "Grosbeak (636 MCM)": {"D": 0.025150, "R25": 0.0001080, "R75": 0.0001290, "Cp": 750.0},
}

# ============================================================
# FUNÇÕES MATEMÁTICAS (IEEE 738)
# ============================================================
MU_F = 1.86e-5       # Viscosidade dinâmica do ar (kg/m s)
RHO_F = 1.16         # Densidade do ar base (kg/m^3)
K_F = 0.026          # Condutividade térmica do ar (W/m K)

def calcular_termico(I, Tc, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75, Cp):
    """Calcula o calor gerado, dissipado e a variação de temperatura (dT) para um instante."""
    pressao_nivel_mar = 101325 
    pressao_local = pressao_nivel_mar * (1 - (2.25577e-5 * elevacao))**5.25588
    rho_ar = (pressao_local * 0.003484) / (Ta + 273.15) 
    
    qs_linear = alpha_abs * qs_global * D
    alpha_R = (R_75 - R_25) / (R_25 * 50) 
    
    # Calor Joule na temperatura atual
    R_ca_tc = R_25 * (1 + alpha_R * (Tc - 25))
    q_joule = (I**2) * R_ca_tc
    
    # Radiação
    q_rad = 17.8 * D * epsilon * (((Tc + 273.15)/100)**4 - ((Ta + 273.15)/100)**4)
    
    # Convecção
    Re = (D * rho_ar * V) / MU_F
    phi = np.radians(angulo_vento)
    K_angle = 1.194 - np.cos(phi) + 0.194 * np.cos(2*phi) + 0.368 * np.sin(2*phi)
    
    if Re < 1000:
        q_c_forcada = K_angle * (1.01 + 1.35 * (Re**0.52)) * K_F * (Tc - Ta)
    else:
        q_c_forcada = K_angle * 0.754 * (Re**0.6) * K_F * (Tc - Ta)
        
    q_c_natural = 3.645 * (rho_ar**0.5) * (D**0.75) * (max(0, Tc - Ta)**1.25)
    q_conv = max(q_c_forcada, q_c_natural)
    
    # Balanço líquido de calor (W/m)
    q_liquido = q_joule + qs_linear - q_conv - q_rad
    
    # Variação de temperatura em 1 segundo (dT = q / Cp)
    dT = q_liquido / Cp
    
    return dT, q_joule, q_conv, q_rad, qs_linear

def simular_transiente(I, Tc_inicial, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75, Cp, T_limite=None, max_tempo=7200):
    """Simula a evolução da temperatura ao longo do tempo (Euler)."""
    Tc = Tc_inicial
    tempo = 0
    tempos = [0]
    temps = [Tc]
    
    dt_step = 1.0 # Passo de 1 segundo
    
    while tempo < max_tempo:
        dT, _, _, _, _ = calcular_termico(I, Tc, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75, Cp)
        
        # Se a variação for quase nula, o cabo atingiu o equilíbrio
        if abs(dT) < 0.0001 and tempo > 10:
            break
            
        Tc += dT * dt_step
        tempo += dt_step
        
        tempos.append(tempo)
        temps.append(Tc)
        
        # Para a simulação se atingir a temperatura limite (ex: 75°C)
        if T_limite and Tc >= T_limite:
            break
            
    return tempos, temps

def calcular_temperatura_estatica(I, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75):
    """Calcula a temperatura de equilíbrio (Regime Permanente)."""
    T_min = Ta 
    T_max = 250.0 
    
    for _ in range(100):
        T_medio = (T_min + T_max) / 2
        dT, _, _, _, _ = calcular_termico(I, T_medio, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75, 1.0)
        
        if dT > 0:
            T_min = T_medio
        else:
            T_max = T_medio
            
        if abs(dT) < 0.0001:
            break
            
    return T_medio

def calcular_ampacidade_estatica(Tc_alvo, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75):
    """Calcula a corrente máxima para uma temperatura alvo."""
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
        return 0.0
        
    return np.sqrt(q_liquido / R_ca_tc)

# ============================================================
# INTERFACE
# ============================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("⚙️ 1. CONFIGURAÇÕES E CLIMA")
    
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
    cp = st.number_input("Capacidade Térmica Cp (J/m·°C)", min_value=1.0, value=cabo_dados["Cp"], format="%.2f", help="Soma da capacidade térmica do alumínio (Outer) e do aço (Core).")
    
    st.markdown("---")
    st.markdown("**Propriedades de Superfície**")
    c3, c4 = st.columns(2)
    with c3:
        alpha_abs = st.number_input("Absortividade", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    with c4:
        epsilon = st.number_input("Emissividade", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

st.markdown("---")

# ============================================================
# ABAS PRINCIPAIS
# ============================================================
aba_estatica, aba_transiente = st.tabs(["📊 Regime Permanente (IEEE 738)", "⏱️ Análise Transiente (Tempo)"])

# ------------------------------------------------------------
# ABA 1: REGIME PERMANENTE
# ------------------------------------------------------------
with aba_estatica:
    modo_calculo = st.radio(
        "O que você deseja calcular?",
        ["Temperatura (dada a Corrente)", "Ampacidade (dada a Temperatura)"],
        horizontal=True
    )
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if modo_calculo == "Temperatura (dada a Corrente)":
            corrente = st.number_input("Corrente CA (A)", min_value=0.0, value=450.0, step=10.0)
            temp_alvo = 90.0
        else:
            temp_alvo = st.number_input("Temperatura Alvo (°C)", min_value=10.0, max_value=150.0, value=90.0, step=5.0)
            corrente = 0.0
            
    if st.button("🔍 Calcular Regime Permanente", type="primary", use_container_width=True):
        
        if modo_calculo == "Temperatura (dada a Corrente)":
            Tc_eq = calcular_temperatura_estatica(corrente, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75)
            st.markdown(f"### Temperatura de Equilíbrio: <span style='color: #00CC96;'>{Tc_eq:.2f} °C</span>", unsafe_allow_html=True)
            I_plot = corrente
            T_plot = Tc_eq
        else:
            I_max = calcular_ampacidade_estatica(temp_alvo, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75)
            st.markdown(f"### Ampacidade Máxima: <span style='color: #00CC96;'>{I_max:.1f} A</span>", unsafe_allow_html=True)
            I_plot = I_max
            T_plot = temp_alvo
            
        # Gráfico
        correntes = np.linspace(0, max(I_plot * 1.5, 500), 40)
        temps = [calcular_temperatura_estatica(c, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75) for c in correntes]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=correntes, y=temps, mode='lines', name='Curva do Cabo', line=dict(color='royalblue', width=3)))
        
        if I_plot > 0 and T_plot < 150:
            fig.add_trace(go.Scatter(x=[I_plot], y=[T_plot], mode='markers+text', name='Ponto Atual', 
                                     text=[f"{T_plot:.1f} °C / {I_plot:.0f} A"], textposition="top center",
                                     marker=dict(color='red', size=12)))
        
        fig.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="Limite 90°C")
        fig.update_layout(title="Curva de Ampacidade", xaxis_title="Corrente CA (A)", yaxis_title="Temperatura (°C)", yaxis=dict(range=[temp_ar, 110]), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# ABA 2: ANÁLISE TRANSIENTE
# ------------------------------------------------------------
with aba_transiente:
    st.markdown("Simule o comportamento térmico do cabo ao longo do tempo. Útil para **sobrecargas de emergência** e **tempo de estabilização**.")
    
    cenario = st.radio(
        "Selecione o Cenário de Simulação:",
        ["Aquecimento (Tempo até o Equilíbrio)", "Sobrecarga (Tempo até 75°C)"],
        horizontal=True
    )
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        if cenario == "Aquecimento (Tempo até o Equilíbrio)":
            I_sim = st.number_input("Corrente de Operação (A)", min_value=0.0, value=450.0, step=10.0)
            Tc_inicial = temp_ar
            T_limite = None
        else:
            I_inicial = st.number_input("Corrente Inicial (A)", min_value=0.0, value=400.0, step=10.0)
            I_sim = st.number_input("Corrente de Emergência (A)", min_value=0.0, value=700.0, step=10.0)
            T_limite = 75.0
            
            # Simula a estabilização com a corrente inicial
            Tc_inicial = calcular_temperatura_estatica(I_inicial, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75)
            st.info(f"O cabo estabilizará em **{Tc_inicial:.2f} °C** com a corrente inicial de {I_inicial} A.")

    with col_t2:
        st.markdown("**Opções de Simulação**")
        max_tempo = st.number_input("Tempo Máximo de Simulação (minutos)", min_value=1, value=120, step=10) * 60
        
    if st.button("⏱️ Simular Transiente", type="primary", use_container_width=True):
        
        tempos, temps = simular_transiente(
            I_sim, Tc_inicial, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, 
            diametro, r25, r75, cp, T_limite, max_tempo
        )
        
        # Resultados
        tempo_final = tempos[-1]
        temp_final = temps[-1]
        
        st.markdown("---")
        st.subheader("🎯 RESULTADO DA SIMULAÇÃO")
        
        m1, m2, m3 = st.columns(3)
        
        if cenario == "Aquecimento (Tempo até o Equilíbrio)":
            m1.metric("Temperatura Final", f"{temp_final:.2f} °C")
            m2.metric("Tempo para Estabilizar", f"{tempo_final/60:.1f} minutos")
            m3.metric("Gradiente Térmico", f"{(temp_final - temp_ar):.1f} °C")
        else:
            if temp_final >= T_limite:
                m1.metric("Temperatura Limite Atingida", f"{temp_final:.2f} °C")
                m2.metric("Tempo de Sobrecarga Segura", f"{tempo_final/60:.1f} minutos")
                m3.metric("Corrente de Emergência", f"{I_sim} A")
                st.error(f"⚠️ O cabo atingiu a temperatura limite de {T_limite}°C. Você tem **{tempo_final/60:.1f} minutos** para aliviar a carga antes de danificar o condutor.")
            else:
                m1.metric("Temperatura Final", f"{temp_final:.2f} °C")
                m2.metric("Tempo de Simulação", f"{tempo_final/60:.1f} minutos")
                m3.metric("Status", "Estável")
                st.success(f"✅ O cabo NÃO atingiu a temperatura limite de {T_limite}°C. A temperatura estabilizou em {temp_final:.2f}°C.")
        
        # Gráfico do Transiente
        fig_t = go.Figure()
        
        # Converte tempo para minutos no eixo X
        tempos_min = [t / 60 for t in tempos]
        
        fig_t.add_trace(go.Scatter(x=tempos_min, y=temps, mode='lines', name='Temperatura do Cabo', line=dict(color='firebrick', width=3)))
        
        if T_limite:
            fig_t.add_hline(y=T_limite, line_dash="dash", line_color="orange", annotation_text=f"Limite ({T_limite}°C)")
            
        fig_t.update_layout(
            title="Curva de Aquecimento Transiente (IEEE 738)",
            xaxis_title="Tempo (minutos)",
            yaxis_title="Temperatura (°C)",
            template="plotly_white",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_t, use_container_width=True)
        
        # Tabela amostral
        with st.expander("📊 Ver Tabela de Dados Amostral (a cada 5 minutos)"):
            df_transiente = pd.DataFrame({"Tempo (min)": tempos_min, "Temperatura (°C)": temps})
            # Filtra a cada 5 minutos (300 segundos)
            df_filtrado = df_transiente[df_transiente["Tempo (min)"] % 5 < 0.02] if len(df_transiente) > 100 else df_transiente
            st.dataframe(df_filtrado.style.format(precision=2), use_container_width=True)
