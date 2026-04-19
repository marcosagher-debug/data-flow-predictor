# Data & Flow Predictor — IMP
## Software Profesional de Aseguramiento de Flujo
### Instituto Mexicano del Petróleo · Posgrado en Ingeniería de Hidrocarburos

---

## 📁 Estructura del Proyecto

```
data_flow_app/
│
├── main_app.py                  ← Aplicación Streamlit principal
├── launcher.py                  ← Punto de entrada para el .exe
├── data_flow_predictor.spec     ← Configuración PyInstaller
├── requirements.txt
├── README.md
│
├── data/                        ← CSVs de entrenamiento (incluidos en el exe)
│   ├── CSV_Parafinas.csv
│   ├── Base_de_Datos_Doctorado_1_co.csv
│   ├── CH4_Metanol__P_T_XCH4O.csv
│   ├── Metanol_y_NaCl_P_t__xch4_y_xNaCl.csv
│   ├── CH4_con_xCH4_.csv
│   └── Base_de_Datos_Doctorado_1_componente.csv
│
├── assets/
│   └── imp_icon.ico             ← Ícono para el ejecutable (proveer)
│
└── ~/.data_flow_predictor/      ← Modelos guardados (fuera del exe)
    models/
    ├── wat_nn.keras
    ├── wat_scaler_0.pkl
    ├── hydrate_nn.keras
    ├── hydrate_scaler_0.pkl
    ├── hybrid_nn.keras
    ├── hybrid_scaler_0.pkl
    └── hybrid_scaler_1.pkl
```

---

## 🚀 Instalación para Desarrollo

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar en modo desarrollo
streamlit run main_app.py
```

---

## 🔨 Compilar como Ejecutable (.exe) — Windows

### Opción A: Comando rápido (--onefile)
```bash
pyinstaller --onefile --windowed \
  --name "DataFlowPredictor" \
  --add-data "data;data" \
  --add-data "assets;assets" \
  --hidden-import streamlit \
  --hidden-import tensorflow \
  --hidden-import sklearn \
  --hidden-import statsmodels \
  --hidden-import openpyxl \
  --hidden-import joblib \
  --icon assets/imp_icon.ico \
  launcher.py
```

### Opción B: Usando el .spec (recomendado, más control)
```bash
# Instalar PyInstaller
pip install pyinstaller

# Compilar
pyinstaller data_flow_predictor.spec

# El ejecutable quedará en: dist/DataFlowPredictor.exe
```

### Opción C: Script automático (Windows)
```bat
@echo off
pip install pyinstaller
pyinstaller data_flow_predictor.spec
echo.
echo ============================================
echo  Ejecutable generado en: dist\DataFlowPredictor.exe
echo ============================================
pause
```

---

## 📱 Uso del Software

1. **Primera ejecución:** Ir a **Dashboard → "Entrenar Todos los Modelos"**
2. **Predicción de Hidratos:** Pestaña "💧 Simulador Hidratos"
3. **Predicción WAT:** Pestaña "🕯️ Simulador Parafinas"
4. **Actualizar modelos:** Pestaña "🔁 Re-Entrenamiento" (subir nuevo CSV)
5. **Exportar:** Pestaña "📄 Reportes" → Excel o PNG

---

## 🧠 Arquitectura de los Modelos

| Modelo | Entradas | Salida | Arquitectura |
|--------|----------|--------|--------------|
| WAT | C1-C7, C8-C15, C16-C23, C24-C30, P, %Par | WAT (°C) | 64→128→32→1, Swish+BN |
| Hidratos CH₄ | T(K) | P(kPa) | 32→16→1, Softplus, log(P) |
| Híbrido | xCH₄O, log_P | T(K) | 128→64→32→1, Swish+BN |

**Preprocesamiento:** RobustScaler en X, log-transform en P(kPa) para hidratos.  
**Persistencia:** Los modelos se guardan en `~/.data_flow_predictor/models/` usando `joblib` y `.keras`.

---

## 📋 Notas para el .exe

- Los modelos entrenados se guardan en `%USERPROFILE%\.data_flow_predictor\` (fuera del exe),
  así mantienen el aprendizaje entre sesiones sin requerir re-compilación.
- Los datos CSV originales se incrustan dentro del ejecutable (`--add-data`).
- TensorFlow puede hacer el .exe pesado (~500 MB). Alternativa: usar `tflite` para producción.

---

## ✅ Dependencias Principales

- Python ≥ 3.10
- TensorFlow ≥ 2.15
- Streamlit ≥ 1.35
- scikit-learn ≥ 1.4
- statsmodels ≥ 0.14
- openpyxl, joblib, matplotlib, seaborn, pandas, numpy

---

*Desarrollado para el IMP — Módulo de IA para Aseguramiento de Flujo*
