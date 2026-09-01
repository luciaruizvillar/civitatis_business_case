import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILOS CSS AVANZADOS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Executive Dashboard - Civitatis",
    page_icon="📊",
    layout="wide"
)

# Inyección de CSS para diseño UI/UX Premium
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Principal */
    .header-banner {
        background: linear-gradient(135deg, #0b192c 0%, #1e293b 100%);
        padding: 30px;
        border-radius: 16px;
        border-left: 10px solid #ff0055;
        color: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        margin-bottom: 30px;
    }
    .header-banner h1 {
        color: #ffffff !important;
        font-weight: 800;
        margin: 0 0 8px 0;
        font-size: 2.2rem;
    }
    .header-banner p {
        color: #94a3b8;
        font-size: 1.1rem;
        margin: 0;
    }

    /* Cards Personalizadas */
    .custom-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0b192c;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8fafc;
        border-radius: 8px 8px 0px 0px;
        gap: 2px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff0055 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CARGA DE DATOS Y MODELADO
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1200
    
    df = pd.DataFrame({
        'lead_time': np.random.randint(0, 60, size=n),
        'importe': np.random.exponential(scale=50, size=n) + 20,
        'personas_num': np.random.randint(1, 6, size=n),
        'canal': np.random.choice(['Directo', 'Email', 'SEO', 'SEM', 'Social'], size=n, p=[0.15, 0.25, 0.3, 0.2, 0.1]),
        'pais': np.random.choice(['España', 'México', 'Argentina', 'Colombia'], size=n),
        'destino': np.random.choice(['París', 'Roma', 'Londres', 'Praga', 'Madrid'], size=n),
        'dispositivo': np.random.choice(['Mobile', 'Desktop'], size=n, p=[0.6, 0.4])
    })
    
    prob_canc = 0.1 + (df['lead_time'] * 0.004) + (df['canal'].isin(['Social', 'SEM']) * 0.08)
    df['cancelado'] = (np.random.rand(n) < prob_canc).astype(int)
    return df

@st.cache_resource
def train_models(data):
    df_model = data.copy()
    
    # Preparación Regresión Logística (Econométrico)
    df_logit = pd.get_dummies(df_model[['cancelado', 'importe', 'personas_num', 'canal']], columns=['canal'], drop_first=False, dtype=float)
    X_logit = df_logit.drop(columns=['cancelado'])
    X_logit = sm.add_constant(X_logit)
    y_logit = df_logit['cancelado']
    
    logit_model = sm.Logit(y_logit, X_logit).fit(disp=0)
    
    # Preparación Random Forest (Predictivo)
    df_rf = pd.get_dummies(df_model[['cancelado', 'lead_time', 'importe', 'personas_num', 'canal', 'dispositivo']], drop_first=True)
    X_rf = df_rf.drop(columns=['cancelado'])
    y_rf = df_rf['cancelado']
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_rf, y_rf)
    
    return logit_model, rf_model, list(X_rf.columns)

df = load_data()
logit_model, rf_model, feature_names = train_models(df)
UMBRAL_OPTIMO = 0.1376

# -----------------------------------------------------------------------------
# 3. HEADER PRINCIPAL
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="header-banner">
        <h1>¿Qué está ocurriendo?</h1>
        <p>Monitor integral del comportamiento operativo, volumen transaccional, fuga por cancelación y optimización de recurrencia.</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. PESTAÑAS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "KPIs & Puente Financiero", 
    "Destinos & Eficiencia", 
    "Clientes & Comportamiento", 
    "Modelo Econométrico & Predictivo", 
    "Plan Estratégico & ROI"
])

# -----------------------------------------------------------------------------
# TAB 1: KPIs & PUENTE FINANCIERO
# -----------------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="custom-card"><div class="metric-label">GMV Bruto</div><div class="metric-value">628.683 €</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="custom-card"><div class="metric-label">Fuga Cancelación</div><div class="metric-value" style="color: #ff0055;">108.741 €</div><small style="color:#ff0055;">🔻 17.30% del GMV</small></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="custom-card"><div class="metric-label">Venta Neta Real</div><div class="metric-value" style="color: #00c853;">519.941 €</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="custom-card"><div class="metric-label">Recurrencia Email</div><div class="metric-value">44,88%</div></div>', unsafe_allow_html=True)

    st.write("---")
    st.subheader("Puente Financiero de Ingresos")
    
    fig_waterfall = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["GMV Bruto", "Fuga por Cancelación", "Venta Neta Real"],
        textposition="outside",
        text=["628.683 €", "-108.741 €", "519.941 €"],
        y=[628683.07, -108741.47, 519941.60],
        connector={"line": {"color": "#94a3b8"}},
        decreasing={"marker": {"color": "#ff0055"}},
        increasing={"marker": {"color": "#00c853"}},
        totals={"marker": {"color": "#0b192c"}}
    ))
    fig_waterfall.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: DESTINOS & EFICIENCIA
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Desglose Operativo por Destino y Canal")
    col_left, col_right = st.columns(2)
    
    with col_left:
        fig_dest = px.histogram(df, x='destino', color='destino', title="Volumen de Reservas por Ciudad", color_discrete_sequence=px.colors.qualitative.Bold)
        fig_dest.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dest, use_container_width=True)
        
    with col_right:
        df_canal = df.groupby('canal')['cancelado'].mean().reset_index()
        df_canal['tasa'] = df_canal['cancelado'] * 100
        fig_canal = px.bar(df_canal, x='canal', y='tasa', color='canal', labels={'tasa': 'Cancelación (%)'}, title="Tasa de Cancelación según Canal", color_discrete_sequence=['#ff0055']*len(df_canal))
        fig_canal.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_canal, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: CLIENTES & COMPORTAMIENTO
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Análisis de Comportamiento e Inferencias Estadísticas")
    ca, cb = st.columns(2)
    
    with ca:
        fig_box = px.box(df, x='pais', y='importe', color='pais', title="Gasto por País Emisor (Medianas e IQR)")
        fig_box.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with cb:
        fig_disp = px.histogram(df, x='pais', color='dispositivo', barmode='group', title="Preferencia Mobile vs Desktop por Mercado")
        fig_disp.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_disp, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: MODELO ECONOMÉTRICO & PREDICTIVO (ESTILO IMAGEN REQUERIDO)
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Análisis Econométrico e Inferencia del Modelo Logit")
    st.write("A continuación se muestra la salida econométrica oficial junto con la estimación visual de los Odds Ratios por canal de adquisición.")
    
    col_tabla, col_forest = st.columns([1.1, 1])
    
    # --- COLUMNA IZQUIERDA: TABLA ECONOMÉTRICA ---
    with col_tabla:
        st.markdown("#### Tabla 4.1: Salida del Modelo Econométrico Logit")
        
        # Extracción de parámetros del modelo Logit real
        summary_df = pd.DataFrame({
            'Coeficiente (β)': logit_model.params,
            'Error Estándar': logit_model.bse,
            'p-value': logit_model.pvalues,
            'Odds Ratio': np.exp(logit_model.params)
        })
        
        # Formato numérico estricto
        summary_formatted = summary_df.style.format({
            'Coeficiente (β)': '{:.4f}',
            'Error Estándar': '{:.4f}',
            'p-value': '{:.4f}',
            'Odds Ratio': '{:.4f}'
        })
        
        st.dataframe(summary_formatted, use_container_width=True, height=380)

    # --- COLUMNA DERECHA: FOREST PLOT (IGUAL A LA IMAGEN) ---
    with col_forest:
        st.markdown("#### Gráfico 4.2: Forest Plot de Impacto por Canal (Odds Ratio)")
        
        params = logit_model.params.drop('const')
        conf = logit_model.conf_int().drop('const')
        odds_ratios = np.exp(params)
        conf_odds = np.exp(conf)
        
        df_forest = pd.DataFrame({
            'Variable': params.index,
            'OR': odds_ratios.values,
            'CI_lower': conf_odds[0].values,
            'CI_upper': conf_odds[1].values
        })

        fig_forest = go.Figure()

        # Línea de referencia (Odds Ratio = 1)
        fig_forest.add_shape(
            type="line", x0=1, x1=1, y0=-0.5, y1=len(df_forest)-0.5,
            line=dict(color="#ff0055", width=2, dash="dash")
        )

        # Añadir barras de error e intervalos
        for idx, row in df_forest.iterrows():
            fig_forest.add_trace(go.Scatter(
                x=[row['CI_lower'], row['OR'], row['CI_upper']],
                y=[row['Variable'], row['Variable'], row['Variable']],
                mode='lines+markers',
                line=dict(color='#ff0055', width=2),
                marker=dict(size=[6, 10, 6], color=['#ff0055', '#0b192c', '#ff0055']),
                showlegend=False
            ))

        fig_forest.update_layout(
            xaxis_title="Odds Ratio (>1 Aumenta Riesgo | <1 Reduce Riesgo)",
            yaxis_title="Variable",
            height=380,
            margin=dict(l=20, r=20, t=20, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0")
        )
        st.plotly_chart(fig_forest, use_container_width=True)

    st.write("---")
    
    # --- SIMULADOR DE RIESGO DE CANCELACIÓN ---
    st.subheader("🔮 Simulador Predictivo en Tiempo Real")
    
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        in_lead = st.slider("Días de Antelación (Lead Time)", 0, 90, 15)
        in_canal = st.selectbox("Canal de Origen", ['Email', 'SEO', 'SEM', 'Social'])
    with sc2:
        in_importe = st.number_input("Importe (€)", 5.0, 2000.0, 75.0, step=5.0)
        in_disp = st.selectbox("Dispositivo", ['Mobile', 'Desktop'])
    with sc3:
        in_personas = st.number_input("Número de Personas", 1, 20, 2)

    if st.button("Evaluar Riesgo de la Reserva", type="primary"):
        input_dict = {col: 0 for col in feature_names}
        if 'lead_time' in input_dict: input_dict['lead_time'] = in_lead
        if 'importe' in input_dict: input_dict['importe'] = in_importe
        if 'personas_num' in input_dict: input_dict['personas_num'] = in_personas
        
        canal_col = f"canal_{in_canal}"
        if canal_col in input_dict: input_dict[canal_col] = 1
        disp_col = f"dispositivo_{in_disp}"
        if disp_col in input_dict: input_dict[disp_col] = 1

        input_df = pd.DataFrame([input_dict])
        prob = rf_model.predict_proba(input_df)[0][1]
        
        st.metric("Probabilidad Estimada de Cancelación", f"{prob:.2%}")
        if prob >= UMBRAL_OPTIMO:
            st.error(f"⚠️ **RESERVA DE ALTO RIESGO** (Supera el umbral óptimo de {UMBRAL_OPTIMO:.2%}). Se recomienda activar flujo preventivo suave vía WhatsApp.")
        else:
            st.success(f"✅ **RESERVA DE BAJO RIESGO** (Por debajo del umbral de {UMBRAL_OPTIMO:.2%}).")

# -----------------------------------------------------------------------------
# TAB 5: PLAN ESTRATÉGICO & ROI (ESTRATEGIAS DE ALTO IMPACTO)
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("🚀 Plan Estratégico de Optimización y ROI Estimado")
    
    p1, p2 = st.columns(2)
    
    with p1:
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #ff0055; margin-top:0;">1. Campaña Cross-Destination Post-Viaje</h4>
            <p><strong>Problema:</strong> El 88.30% de los clientes son Mono-Destino.</p>
            <p><strong>Acción:</strong> Disparar flujos dinámicos de Email Marketing 7 días después del tour con sugerencias personalizadas para las siguientes capitales clave.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #ff0055; margin-top:0;">2. Reestructuración de Comisiones a Afiliados</h4>
            <p><strong>Problema:</strong> Elevada tasa de cancelación originada en tráfico pagado/afiliados.</p>
            <p><strong>Acción:</strong> Modificar los contratos de afiliados pagando la comisión únicamente si la reserva no es cancelada en los 14 días posteriores a la compra.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with p2:
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #ff0055; margin-top:0;">3. Optimización UX en Checkout Móvil</h4>
            <p><strong>Problema:</strong> La conversión Desktop (5.15%) duplica la conversión Móvil (2.33%).</p>
            <p><strong>Acción:</strong> Integración de pasarelas 1-Click (Apple Pay / Google Pay) y simplificación del formulario de viajeros en smartphone.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #ff0055; margin-top:0;">4. Gestión Activa de Reservas con Lead Time > 15 días</h4>
            <p><strong>Problema:</strong> A mayor antelación, mayor tasa de arrepentimiento/cancelación.</p>
            <p><strong>Acción:</strong> Notificaciones automatizadas de confirmación flexible que incentiven el cambio de fecha en lugar de la cancelación total.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    st.subheader("💰 Retorno de Inversión (ROI Proyectado)")
    
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Recuperación de Cancelaciones (15%)", "+16.311 €", "Impacto Directo")
    with r2:
        st.metric("Ventas Adicionales por Cross-Sell (+5%)", "+25.997 €", "Fidelización")
    with r3:
        st.metric("Impacto Total Proyectado en EBITDA", "+42.308 €", "Beneficio Neto Estimado")