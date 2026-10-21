import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import io

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
st.caption("Cálculo térmico de regime permanente, transitório e geração de relatórios.")
st.markdown("---")

# ============================================================
# BANCO DE DADOS DE CABOS (ESPAÇO RESERVADO)
# ============================================================
CABOS = {
    "Linnet (336,4 MCM)": {
        "D": 0.018288, 
        "R25": 0.0001695, 
        "R75": 0.0002028,
        "Cp": 552.81  # 449.771 (Outer) + 103.039 (Core)
    },
    "Grosbeak (636 MCM)": {
        "D": 0.025146,
        "R25": 0.0000902,
        "R75": 0.0001077,
        "Cp": 1045.981 # 850.957 (Outer) + 195.024 (Core)
    },
    # Adicione novos cabos aqui no futuro
}

# ============================================================
# FUNÇÕES MATEMÁTICAS (IEEE 738)
# ============================================================
MU_F = 1.86e-5       
RHO_F = 1.16         
K_F = 0.026          

def calcular_termico(I, Tc, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75, Cp):
    pressao_nivel_mar = 101325 
    pressao_local = pressao_nivel_mar * (1 - (2.25577e-5 * elevacao))**5.25588
    rho_ar = (pressao_local * 0.003484) / (Ta + 273.15) 
    
    qs_linear = alpha_abs * qs_global * D
    alpha_R = (R_75 - R_25) / (R_25 * 50) 
    
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
        
    q_c_natural = 3.645 * (rho_ar**0.5) * (D**0.75) * (max(0, Tc - Ta)**1.25)
    q_conv = max(q_c_forcada, q_c_natural)
    
    q_liquido = q_joule + qs_linear - q_conv - q_rad
    dT = q_liquido / Cp
    return dT

def simular_transiente(I, Tc_inicial, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75, Cp, T_limite=None, max_tempo=14400):
    Tc = Tc_inicial
    tempo = 0
    tempos = [0]
    temps = [Tc]
    dt_step = 1.0 
    
    while tempo < max_tempo:
        dT = calcular_termico(I, Tc, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75, Cp)
        
        if abs(dT) < 0.0001 and tempo > 10:
            break
            
        Tc += dT * dt_step
        tempo += dt_step
        tempos.append(tempo)
        temps.append(Tc)
        
        if T_limite and Tc >= T_limite:
            break
            
    return tempos, temps

def calcular_temperatura_estatica(I, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75):
    T_min, T_max = Ta, 250.0 
    for _ in range(100):
        T_medio = (T_min + T_max) / 2
        dT = calcular_termico(I, T_medio, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75, 1.0)
        
        if dT > 0: T_min = T_medio
        else: T_max = T_medio
            
        if abs(dT) < 0.0001: break
    return T_medio

def calcular_ampacidade_estatica(Tc_alvo, Ta, V, angulo_vento, qs_global, alpha_abs, epsilon, elevacao, D, R_25, R_75):
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
    
    if q_liquido <= 0: return 0.0
    return np.sqrt(q_liquido / R_ca_tc)

# ============================================================
# INTERFACE DE ENTRADA
# ============================================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("⚙️ 1. CONFIGURAÇÕES E CLIMA")
    c1, c2 = st.columns(2)
    with c1:
        temp_ar = st.number_input("Temp. do Ar (°C)", value=40.0, step=1.0)
        vel_vento = st.number_input("Vel. do Vento (m/s)", min_value=0.0, value=0.61, step=0.1)
        elevacao = st.number_input("Elevação (m)", value=0.0, step=50.0)
    with c2:
        ang_vento = st.number_input("Ângulo do Vento (°)", min_value=0.0, max_value=90.0, value=90.0, step=10.0)
        rad_solar = st.number_input("Radiação Solar (W/m²)", min_value=0.0, value=1200.0, step=100.0)
        
with col2:
    st.subheader("🔌 2. PROPRIEDADES DO CONDUTOR")
    cabo_nome = st.selectbox("Selecione o Cabo", list(CABOS.keys()))
    cabo_dados = CABOS[cabo_nome]
    
    diametro = st.number_input("Diâmetro Externo D (m)", value=cabo_dados["D"], format="%.6f")
    r25 = st.number_input("R CA a 25°C (Ω/m)", value=cabo_dados["R25"], format="%.7f")
    r75 = st.number_input("R CA a 75°C (Ω/m)", value=cabo_dados["R75"], format="%.7f")
    cp = st.number_input("Capacidade Térmica Cp (J/m·°C)", value=cabo_dados["Cp"], format="%.2f")
    
    st.markdown("---")
    st.markdown("**Propriedades de Superfície**")
    c3, c4 = st.columns(2)
    with c3: alpha_abs = st.number_input("Absortividade", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    with c4: epsilon = st.number_input("Emissividade", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

st.markdown("---")

# ============================================================
# CRIAÇÃO DAS ABAS
# ============================================================
aba_estatica, aba_transiente = st.tabs(["📊 Regime Permanente", "⏱️ Análise Transiente"])

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
            corrente = st.number_input("Corrente CA (A)", min_value=0.0, value=400.0, step=10.0)
            temp_alvo = 90.0
        else:
            temp_alvo = st.number_input("Temperatura Alvo (°C)", min_value=10.0, max_value=150.0, value=75.0, step=5.0)
            corrente = 0.0
            
    if st.button("🔍 Calcular Regime Permanente", type="primary", use_container_width=True):
        if modo_calculo == "Temperatura (dada a Corrente)":
            Tc_eq = calcular_temperatura_estatica(corrente, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75)
            st.markdown(f"### Temperatura de Equilíbrio: <span style='color: #00CC96;'>{Tc_eq:.2f} °C</span>", unsafe_allow_html=True)
            I_plot, T_plot = corrente, Tc_eq
        else:
            I_max = calcular_ampacidade_estatica(temp_alvo, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75)
            st.markdown(f"### Ampacidade Máxima: <span style='color: #00CC96;'>{I_max:.1f} A</span>", unsafe_allow_html=True)
            I_plot, T_plot = I_max, temp_alvo
            
        correntes = np.linspace(0, max(I_plot * 1.5, 500), 40)
        temps = [calcular_temperatura_estatica(c, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75) for c in correntes]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=correntes, y=temps, mode='lines', name='Curva do Cabo', line=dict(color='royalblue', width=3)))
        if I_plot > 0 and T_plot < 150:
            fig.add_trace(go.Scatter(x=[I_plot], y=[T_plot], mode='markers+text', name='Ponto Atual', text=[f"{T_plot:.1f} °C"], textposition="top center", marker=dict(color='red', size=12)))
        
        fig.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="Limite 90°C")
        fig.update_layout(title="Curva de Ampacidade", xaxis_title="Corrente CA (A)", yaxis_title="Temperatura (°C)", yaxis=dict(range=[temp_ar, 110]), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # FUNÇÃO DE RELATÓRIO PLS-CADD E PDF
    # ========================================================
    st.markdown("---")
    st.subheader("📑 Relatório Completo de Ampacidade")
    st.write("Gera uma tabela de 25°C a 75°C (passo de 0,5°C) com e sem radiação solar, no formato PLS-CADD.")

    if st.button("📄 Gerar Relatório (Estilo PLS-CADD)", type="secondary", use_container_width=True):
        
        # 1. Gera os dados da tabela
        temps_tab = np.arange(25.0, 75.5, 0.5)
        dados = []
        for t in temps_tab:
            i_c_sol = calcular_ampacidade_estatica(t, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75)
            i_s_sol = calcular_ampacidade_estatica(t, temp_ar, vel_vento, ang_vento, 0.0, alpha_abs, epsilon, elevacao, diametro, r25, r75)
            dados.append([t, i_c_sol, i_s_sol, i_c_sol - i_s_sol])
            
        df_rel = pd.DataFrame(dados, columns=["Temperatura (°C)", "Ampacidade c/ Sol (A)", "Ampacidade s/ Sol (A)", "Ganho Solar (A)"])

        # 2. Monta o texto do Memorial (Idêntico ao PLS)
        memorial_texto = f"""IEEE Standard 738-2006 method of calculation
------------------------------------------------------------
Weather Conditions
Air temperature:                    {temp_ar:6.3f}  (deg C)
Wind speed:                         {vel_vento:6.3f}  (m/s)
Wind to conductor angle:            {ang_vento:6.3f}  (deg)

Solar Conditions
Measured solar radiation:           {rad_solar:6.3f}  (Watt/m^2)

Conductor Properties
Description: {cabo_nome}
AC resistance at 25.0 (deg C):      {r25*1000:6.4f}  (Ohm/km)
AC resistance at 75.0 (deg C):      {r75*1000:6.4f}  (Ohm/km)
Solar absorptivity:                 {alpha_abs:6.3f}
Emissivity:                         {epsilon:6.3f}
Outer diameter:                     {diametro*1000:6.3f}  (mm)
Elevation above sea level:          {elevacao:6.3f}  (m)
"""
        
        st.markdown("### 📝 Memorial Descritivo")
        st.code(memorial_texto, language="text")

        st.markdown("### 📊 Tabela de Ampacidade")
        st.dataframe(df_rel.style.format({
            "Temperatura (°C)": "{:.1f}",
            "Ampacidade c/ Sol (A)": "{:.2f}",
            "Ampacidade s/ Sol (A)": "{:.2f}",
            "Ganho Solar (A)": "{:.2f}"
        }), use_container_width=True, height=400)

        # 3. Gera o PDF
        def gerar_pdf():
            pdf = FPDF()
            pdf.add_page()
            
            # Título
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, txt="Relatorio de Ampacidade - IEEE 738-2006", ln=True, align='C')
            pdf.ln(5)
            
            # Memorial
            pdf.set_font("Courier", size=9)
            for linha in memorial_texto.split('\n'):
                pdf.cell(0, 4, txt=linha, ln=True)
            
            pdf.ln(10)
            
            # Tabela
            pdf.set_font("Arial", 'B', 10)
            col_widths = [40, 45, 45, 45]
            headers = ["Temp (C)", "c/ Sol (A)", "s/ Sol (A)", "Ganho (A)"]
            
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 8, header, border=1, align='C')
            pdf.ln()
            
            pdf.set_font("Arial", size=9)
            for index, row in df_rel.iterrows():
                # Negrito a cada 5 graus (25, 30, 35...)
                if row["Temperatura (°C)"] % 5.0 == 0:
                    pdf.set_font("Arial", 'B', 9)
                else:
                    pdf.set_font("Arial", size=9)
                    
                pdf.cell(col_widths[0], 6, f"{row['Temperatura (°C)']:.1f}", border=1, align='C')
                pdf.cell(col_widths[1], 6, f"{row['Ampacidade c/ Sol (A)']:.2f}", border=1, align='C')
                pdf.cell(col_widths[2], 6, f"{row['Ampacidade s/ Sol (A)']:.2f}", border=1, align='C')
                pdf.cell(col_widths[3], 6, f"{row['Ganho Solar (A)']:.2f}", border=1, align='C')
                pdf.ln()

            return pdf.output(dest="S").encode("latin-1")

        try:
            pdf_bytes = gerar_pdf()
            st.download_button(
                label="📥 Baixar Relatório em PDF",
                data=pdf_bytes,
                file_name=f"relatorio_ampacidade_{cabo_nome.split(' ')[0]}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Erro ao gerar PDF: {e}")

# ------------------------------------------------------------
# ABA 2: ANÁLISE TRANSIENTE
# ------------------------------------------------------------
with aba_transiente:
    st.markdown("Simule o comportamento térmico do cabo ao longo do tempo.")
    
    cenario = st.radio(
        "Selecione o Cenário:",
        ["Aquecimento (Tempo até o Equilíbrio)", "Sobrecarga (Tempo até 75°C)"],
        horizontal=True
    )
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        if cenario == "Aquecimento (Tempo até o Equilíbrio)":
            I_sim = st.number_input("Corrente de Operação (A)", min_value=0.0, value=450.0, step=10.0, key="i_aquec")
            Tc_inicial = temp_ar
            T_limite = None
            st.info(f"O cabo começará na temperatura ambiente ({Tc_inicial:.1f}°C) e aquecerá com {I_sim} A.")
        else:
            I_inicial = st.number_input("Corrente Atual (Regime Permanente) (A)", min_value=0.0, value=400.0, step=10.0, key="i_ini")
            I_sim = st.number_input("Corrente de Emergência (Sobrecarga) (A)", min_value=0.0, value=700.0, step=10.0, key="i_emerg")
            T_limite = 75.0
            
            Tc_inicial = calcular_temperatura_estatica(I_inicial, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75)
            st.warning(f"**Ponto de Partida:** O cabo está estabilizado em **{Tc_inicial:.2f} °C** operando com {I_inicial} A. De repente, a corrente salta para {I_sim} A.")

    with col_t2:
        st.markdown("**Informações da Simulação**")
        st.write("- A simulação rodará automaticamente até atingir o equilíbrio ou o limite de 75°C.")
        st.write("- Não é necessário definir o tempo máximo.")

    if st.button("⏱️ Simular Transiente", type="primary", use_container_width=True):
        
        tempos, temps = simular_transiente(
            I_sim, Tc_inicial, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, 
            elevacao, diametro, r25, r75, cp, T_limite
        )
        
        tempo_final = tempos[-1]
        temp_final = temps[-1]
        
        st.markdown("---")
        st.subheader("🎯 RESULTADO")
        
        m1, m2, m3 = st.columns(3)
        
        if cenario == "Aquecimento (Tempo até o Equilíbrio)":
            m1.metric("Temperatura Final", f"{temp_final:.2f} °C")
            m2.metric("Tempo para Estabilizar", f"{tempo_final/60:.1f} minutos")
            m3.metric("Gradiente Térmico", f"{(temp_final - temp_ar):.1f} °C")
        else:
            if temp_final >= T_limite - 0.1: 
                m1.metric("Temperatura Limite", f"{temp_final:.2f} °C")
                m2.metric("Tempo de Sobrecarga Segura", f"{tempo_final/60:.2f} minutos")
                m3.metric("Corrente de Emergência", f"{I_sim} A")
                st.error(f"⚠️ O cabo partiu de {Tc_inicial:.1f}°C e atingiu a temperatura limite de {T_limite}°C. Você tem **{tempo_final/60:.1f} minutos** para aliviar a carga antes de danificar o condutor.")
            else:
                m1.metric("Temperatura Final", f"{temp_final:.2f} °C")
                m2.metric("Tempo de Simulação", f"{tempo_final/60:.1f} minutos")
                m3.metric("Status", "Estável (Não atingiu o limite)")
                st.success(f"✅ O cabo NÃO atingiu {T_limite}°C. Mesmo com a sobrecarga de {I_sim} A, a temperatura estabilizou em {temp_final:.2f}°C.")
        
        # Gráfico do Transiente
        fig_t = go.Figure()
        tempos_min = [t / 60 for t in tempos]
        
        fig_t.add_trace(go.Scatter(x=tempos_min, y=temps, mode='lines', name=f'Corrente = {I_sim} A', line=dict(color='firebrick', width=3)))
        
        if T_limite:
            fig_t.add_hline(y=T_limite, line_dash="dash", line_color="orange", annotation_text=f"Limite ({T_limite}°C)")
            fig_t.add_hline(y=Tc_inicial, line_dash="dot", line_color="gray", annotation_text=f"Partida ({Tc_inicial:.1f}°C)")
            
        fig_t.update_layout(title="Curva de Aquecimento Transiente", xaxis_title="Tempo (minutos)", yaxis_title="Temperatura (°C)", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_t, use_container_width=True)
