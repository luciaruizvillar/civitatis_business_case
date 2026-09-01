# Business Case Civitatis — Repetición, Destinos y Estado del Negocio

Análisis del comportamiento de clientes de Civitatis a partir de eventos de Google Analytics, reservas, clientes, tours y proveedores, con el objetivo de responder a las tres preguntas del COMEX: qué explica la repetición de clientes, qué destinos retienen mejor, y cuánto ha vendido realmente el negocio.

## Estructura del repositorio

```
├── codigo_business_case.py          # Análisis: Fase 1 (limpieza e identidad) y Fase 2 (métricas, modelos, tests)
├── aplicacion10.py                  # Aplicación interactiva (Streamlit)
└── README.md                        # Este documento (incluye enlaces de descarga de los CSV)
```

## Datos

Los CSV no están incluidos en el repositorio — no se suben datos a GitHub, y el fichero de eventos supera los 100 MB (límite estándar de GitHub). Descárgalos desde el enlace compartido:

📂 **Descarga de los datasets**: https://drive.google.com/drive/folders/1ewAfZKXZh7EH5-7TlsNPkkfjV2NQX11H?usp=sharing

- `ga_eventos.csv` (~700.000 filas)
- `reservas.csv`
- `clientes.csv`
- `tours.csv`
- `proveedores.csv`



```bash
pip install pandas numpy streamlit plotly statsmodels scikit-learn seaborn matplotlib

# 1. Análisis (notebook/script de limpieza, métricas y modelos)
python codigo_business_case.py        # o el .ipynb equivalente

# 2. Aplicación interactiva
streamlit run aplicacion10.py
```

> Tanto el código del análisis estadístico como el de la aplicación interactiva se desarrollaron originalmente en **Google Colab**, con ayuda de IA para la generación del código (ver sección "Decisiones y uso de IA"). Se han adaptado/exportado a scripts `.py` para poder ejecutarse en local con un solo comando, tal y como pide el enunciado.


---

## Decisiones y uso de IA

### 1. Supuestos adoptados

El enunciado es deliberadamente incompleto; estas son las hipótesis explícitas bajo las que se sostienen las conclusiones:

- **Fuente de verdad para ventas y canal**: `reservas.csv`. Los eventos de GA (`ga_eventos.csv`) se usan solo para navegación y dispositivo, nunca para calcular importes o estados de reserva, porque solo 310 de las 8.414 reservas confirmadas tienen un evento de compra que las respalde (8.105 reservas sin rastro en la analítica web, y 1 evento de compra sin reserva asociada). Cruzar ambas fuentes como si fueran equivalentes habría subestimado drásticamente el negocio real.
- **Normalización de estados**: `CONFIRMADA`, `Cancelada` y `cancelled` se tratan como variantes de dos únicos estados (`confirmada`/`completada` y `cancelada`). Sin fusionar `cancelled` → `cancelada`, la tasa de cancelación real se habría infravalorado (795 vs. 1.160 reservas canceladas).
- **Resolución de identidad**: se cruzan `cookie_id` → `user_id` a partir de eventos donde el `user_id` no es nulo, generando 2.153 pares únicos sin conflictos de dispositivo compartido. Esto eleva la cobertura de usuarios identificados en eventos del 9,91% al 11,89%. Se asume que un `cookie_id` pertenece a un único usuario cuando aparece asociado a un solo `user_id` en el histórico; los `cookie_id` sin ningún evento de login/compra quedan como anónimos y no se imputan.
- **Reservas con `importe_eur` ≤ 0€ (1.297 reservas, 15,4%)**: se asume que corresponden mayoritariamente a Free Tours legítimos (`precio_por_persona_eur = 0` en `tours.csv`), no a errores sistemáticos. No se excluyen del cálculo de GMV, ya que representan actividad real de la plataforma, pero se señalan como una fuente de ruido que el COMEX debe validar operativamente.
- **Reservas con `personas = 0` (15 reservas)**: se mantienen en el dataset pero se excluyen de cualquier métrica que dependa del tamaño de grupo (gasto por persona, etc.), al no poder determinarse si es un error de captura o un caso válido (p. ej. reserva corporativa sin pasajeros nominados).
- **Atribución de campaña incompleta (61% de reservas sin `campana`)**: se excluye del análisis cuantitativo de ROI por campaña. Se acepta hablar de qué **canal** retiene mejor, pero no de qué **campaña** concreta, para evitar extrapolar sobre una submuestra no representativa.
- **Pruebas por país/dispositivo apoyadas en muestra pequeña**: solo 157 de 8.414 reservas tienen país de IP identificado vía el cruce reservas↔eventos de compra. Los resultados de Kruskal-Wallis y Chi-cuadrado por país se presentan como señal direccional, no como verdad estadística consolidada.
- **Proveedores con dirección inconsistente** (p. ej. "Tours Roma Group 1" con sede en Sevilla): se interpreta como ruido propio de datos sintéticos y se documenta como hallazgo de control de calidad, sin corregirlo, ya que no afecta a las métricas de venta o recurrencia.

### 2. Definición exacta de cada métrica clave

| Métrica | Definición | Justificación |
|---|---|---|
| **Venta (GMV Bruto)** | Suma de `importe_eur` de reservas en estado `confirmada` o `completada`. | Refleja el compromiso de compra tal como queda registrado en el sistema transaccional, antes de descontar el churn operativo. |
| **Venta Neta Real** (métrica propia) | `GMV Bruto − Fuga por cancelación`, donde la fuga es la suma de `importe_eur` de reservas en estado `cancelada`. | El GMV bruto sobrestima el negocio en un 17,3%; la venta neta es la cifra que se defiende ante el COMEX como "cuánto hemos vendido realmente". |
| **Sesión** | Registro identificado por `session_id` en `ga_eventos.csv`, independientemente de si el usuario está identificado (`user_id`) o es anónimo (`cookie_id`/`temp_client_id`). | Es la unidad mínima de comportamiento de navegación disponible en los datos. |
| **Cliente recurrente** | `user_id` con **2 o más reservas** en estado distinto de `cancelada` (`reservas_validas`). | Se excluyen las canceladas para no contar como "repetición" un intento de compra que nunca se materializó. |
| **Conversión (por dispositivo/sesión)** | `nº de sesiones con reserva_id no nulo / nº total de sesiones`, tras normalizar `device` (fusionando duplicados como `desktp` → `desktop`, `Desktop`/`desktop` → `desktop`). | Sin esta limpieza, la inconsistencia de mayúsculas y typos en `device` fragmenta artificialmente el dato y distorsiona la comparación Desktop vs. Mobile. |
| **Tasa de cancelación** | `nº reservas con estado_clean = 'cancelada' / nº total de reservas`, tras normalizar `cancelled` → `cancelada`. | Ver punto anterior sobre normalización de estados. |
| **Cliente multi-destino** | `user_id` con `nunique(destino)` > 1 sobre reservas válidas (destino extraído de la URL del tour). | Usada para medir el efecto de la diversificación de destino sobre la recurrencia (r = 0,68 con total de reservas). |

### 3. Qué tareas se delegaron en IA

- **Estructuración del trabajo por fases** (Fase 1: auditoría de datos, resolución de identidad y limpieza; Fase 2: definición de métricas, modelos y visualizaciones): se usó IA para organizar y secuenciar el análisis de forma más clara, partiendo de las preguntas del COMEX ya priorizadas manualmente.
- **Generación de código** (Gemini): a partir de la estructura y los estadísticos que yo decidí de antemano (qué cruces hacer, qué tests aplicar, qué normalizar), la IA generó el código Python (pandas, statsmodels, scikit-learn, seaborn/plotly) para ejecutarlos. Todas las salidas (tablas, gráficos, resultados de tests) fueron revisadas por mí antes de incorporarlas a las conclusiones.
- **Boceto de la presentación** (Claude): partiendo de un storytelling definido por mí (arco narrativo: contexto emocional → estado del negocio → calidad del dato → motor de la repetición → destinos → plan de acción), la IA generó un boceto de contenidos y estructura de slides, que posteriormente se maquetó y refinó manualmente en Canva para la versión final del deck.
- **Redacción de interpretaciones ejecutivas** de cada salida estadística (p. ej. lectura de odds ratios, interpretación de la matriz de correlación de Pearson, lectura de la descomposición estacional): generadas por IA a partir de los resultados numéricos ya calculados, y verificadas contra los propios números antes de usarlas.

### 4. Qué propuso la IA y se descartó, y por qué

- **Modelo de Random Forest para predecir cancelaciones** (uplift táctico, con `lead_time`, `importe_eur`, mes, día de la semana): se probó, pero se descartó como herramienta operativa de producción. Con el umbral por defecto (0,5) el modelo tenía Recall = 0 en la clase minoritaria (ilusión de accuracy del 86%). Ajustando `class_weight='balanced'` y buscando el umbral óptimo por F1-score (0,1376), el mejor resultado alcanzado fue Recall = 57% / Precisión = 20% con ROC-AUC = 0,61 — apenas por encima del azar. Se documenta como línea de trabajo futura (faltan variables explicativas: histórico de cancelaciones del usuario, tipo de tour, antelación de pago, etc.), pero no se presenta al COMEX como modelo accionable.
- **Mapa geoespacial de burbujas por país de IP**: se descartó como pieza central del deck. Solo hay coordenadas mapeadas para 6 países y la cobertura de país de IP en reservas es muy baja (157 de 8.414, vía el cruce con GA), por lo que el mapa habría dado una falsa sensación de cobertura global. Se mantiene como material exploratorio en el notebook, no en el memo ejecutivo.
- **ROI por campaña de marketing**: la IA propuso desglosar el impacto por `campana`, pero se descartó al detectar que el 61% de las reservas no tiene campaña registrada; cualquier cifra de ROI por campaña habría sido una extrapolación no soportada sobre un 39% de la muestra.
- **Uso de test paramétricos (t-Student/ANOVA)** para comparar gasto entre grupos: descartado tras el test de Shapiro-Wilk sobre `importe_eur` (p = 7,71×10⁻⁵⁰), que rechaza la normalidad de forma contundente (sesgo por Free Tours a 0€ y reservas grupales). Se optó por los no paramétricos equivalentes (Kruskal-Wallis, Chi-cuadrado) y por reportar medianas/IQR en lugar de medias.

### 5. Preguntas de aclaración resueltas por hipótesis propia (no enviadas por email)

- **Formato de entrega al COMEX**: el enunciado permite "memo ejecutivo o deck de máximo 5 diapositivas". Se asume que un deck visual, apoyado en storytelling, comunica mejor a un comité que no quiere ver datos que un memo de texto corrido, y que puede sostenerse solo sin necesidad de la aplicación (tal y como pide el punto 4 de entregables). Se opta por el deck en lugar de pedir aclaración sobre el formato preferido.
- **Herramienta para la aplicación interactiva**: el enunciado permite elegir libremente la herramienta ("Streamlit, Dash, Shiny, Evidence…") y menciona explícitamente que la app debe ser "ejecutable en local con un solo comando". Se asume que **Tableau no encaja** en ese requisito: Tableau construye dashboards mediante su propia interfaz visual y los publica/comparte vía Tableau Public o Tableau Server, pero no se ejecuta ni se versiona como una aplicación de código con un único comando de arranque (`streamlit run ...`) desde un repositorio Git. Como el entregable pide explícitamente código versionado con historial de commits y un comando de ejecución local, se descarta Tableau desde el inicio y se opta por Streamlit sin plantear la duda por email, al considerarse una decisión de herramienta y no una ambigüedad del enunciado.

---

## Hallazgos priorizados (resumen — ver deck para el detalle)

1. **Estado del negocio**: GMV bruto de 628.683€, con una fuga por cancelación de 108.741€ (13,79% de las reservas), dejando una Venta Neta Real de 519.942€. La fuga no es aleatoria: la tasa de cancelación pasa de 6,25% (mismo día) a 18,68% (+30 días de antelación), y el 58,5% del dinero cancelado proviene de reservas con más de 16 días de antelación.
2. **Repetición**: Email es el canal que más fideliza (44,9% de recurrencia) pese a ser uno de los que menos volumen mueve; también es el que menos riesgo de cancelación presenta (OR = 0,63 frente a Afiliados). El motor real de la recurrencia es la diversificación de destino (r = 0,68 con nº de reservas), no el precio ni la antigüedad de cuenta (r ≈ 0).
3. **Destinos**: París, Roma, Madrid y Londres concentran más del 60% del volumen, pero no son los que mejor retienen — Madrid y París combinan alto volumen con la mayor tasa de cancelación (~14,9%). Roma es el caso de éxito: alto volumen con la cancelación más baja del top 4 (12,3%).
