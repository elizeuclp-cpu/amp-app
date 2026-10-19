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
            I_sim = st.number_input("Corrente de Operação (A)", min_value=0.0, value=450.0, step=10.0, key="i_aquec")
            Tc_inicial = temp_ar
            T_limite = None
        else:
            I_inicial = st.number_input("Corrente Inicial (A)", min_value=0.0, value=400.0, step=10.0, key="i_ini")
            I_sim = st.number_input("Corrente de Emergência (A)", min_value=0.0, value=700.0, step=10.0, key="i_emerg")
            T_limite = 75.0
            
            # Simula a estabilização com a corrente inicial
            Tc_inicial = calcular_temperatura_estatica(I_inicial, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, diametro, r25, r75)
            st.info(f"Temperatura de equilíbrio para {I_inicial} A: **{Tc_inicial:.2f} °C**")

    with col_t2:
        st.markdown("**Opções de Simulação**")
        max_tempo = st.number_input("Tempo Máximo de Simulação (minutos)", min_value=1, value=120, step=10, key="max_t") * 60
        
    if st.button("⏱️ Simular Transiente", type="primary", use_container_width=True, key="btn_simular"):
        
        # 1. Roda a simulação para o cenário escolhido
        tempos, temps = simular_transiente(
            I_sim, Tc_inicial, temp_ar, vel_vento, ang_vento, rad_solar, alpha_abs, epsilon, elevacao, 
            diametro, r25, r75, cp, T_limite, max_tempo
        )
        
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
                st.error(f"⚠️ O cabo atingiu a temperatura limite de {T_limite}°C. Você tem **{tempo_final/60:.1f} minutos** para aliviar a carga.")
            else:
                m1.metric("Temperatura Final", f"{temp_final:.2f} °C")
                m2.metric("Tempo de Simulação", f"{tempo_final/60:.1f} minutos")
                m3.metric("Status", "Estável")
                st.success(f"✅ O cabo NÃO atingiu a temperatura limite de {T_limite}°C. A temperatura estabilizou em {temp_final:.2f}°C.")
        
        # Gráfico do Transiente
        fig_t = go.Figure()
        tempos_min = [t / 60 for t in tempos]
        fig_t.add_trace(go.Scatter(x=tempos_min, y=temps, mode='lines', name=f'Corrente = {I_sim} A', line=dict(color='firebrick', width=3)))
        
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
