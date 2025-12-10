# Trabajo Individual N2 - IA Aplicada: Machine Learning
## Predicción de Niveles de Ingresos - Análisis de Censo Poblacional

---

## 🎉 ESTADO DEL PROYECTO

**✅ NOTEBOOKS CREADOS Y LISTOS PARA USAR**

Los 5 notebooks ya están creados en la carpeta `notebooks/` con su estructura completa:

1. **01_EDA_Completo.ipynb** → Pregunta 3 (Análisis Exploratorio)
2. **02_Preprocesamiento.ipynb** → Pregunta 4 (División de datos)
3. **03_Modelo1_RegresionLogistica.ipynb** → Pregunta 5 (Modelo 1)
4. **04_Modelo2_RegresionLineal.ipynb** → Pregunta 5 (Modelo 2)
5. **05_Comparacion_Validacion_Final.ipynb** → Pregunta 6 (Validación)

📝 **Ver `GUIA_IMPLEMENTACION.md` para el código completo de cada celda**

---

## 📊 RESUMEN EJECUTIVO

Este proyecto tiene como objetivo **predecir la propensión de los niveles de ingresos** de habitantes a partir de datos de una encuesta nacional de hogares y salud familiar. Se utilizará la metodología **CRISP-DM** y se construirán **modelos de Machine Learning** para identificar poblaciones vulnerables y apoyar decisiones de políticas públicas.

### Datos Principales:
- **Registros:** 32,561 personas
- **Variables:** 10 columnas (5 numéricas, 5 categóricas)
- **Variable Objetivo:** INGRESO (<=50K, >50K)
- **Fuente:** Encuesta nacional de hogares realizada por ONG asociada al gobierno

---

## 📝 RESPUESTAS A PREGUNTAS 1-4

### **1. DEFINICIÓN DEL PROBLEMA Y OBJETIVOS DE NEGOCIO (02 puntos)**

#### **Naturaleza del Problema:**

Este es un problema de **CLASIFICACIÓN BINARIA** en el contexto de análisis socioeconómico. Aunque el PDF solicita modelos de regresión lineal, la variable objetivo INGRESO es **categórica** con dos clases:
- `<=50K`: Ingreso menor o igual a $50,000
- `>50K`: Ingreso mayor a $50,000

La problemática central es identificar patrones sociodemográficos y económicos que permitan predecir si una persona pertenece a un grupo de ingresos bajos o altos, con el fin de focalizar políticas públicas en poblaciones vulnerables.

#### **Contexto del Problema:**

La ONG asociada al gobierno necesita:
1. Categorizar habitantes según su ingreso total
2. Estudiar el indicador de % de pobreza
3. Identificar poblaciones vulnerables
4. Priorizar grupos de mayor necesidad para proyectos de inversión
5. Diseñar políticas públicas para mitigar la pobreza

#### **Objetivos de Negocio:**

**Objetivo General:**
- Predecir la propensión de los niveles de ingresos de los habitantes basándose en características sociodemográficas y educativas.

**Objetivos Específicos:**

1. **Identificación de Poblaciones Vulnerables:**
   - Determinar qué características (edad, educación, horas trabajadas, etc.) están asociadas con ingresos ≤$50K
   - Identificar grupos de mayor riesgo de pobreza

2. **Optimización de Recursos Públicos:**
   - Proporcionar herramientas predictivas para focalizar inversión pública
   - Priorizar intervenciones en grupos de mayor necesidad

3. **Análisis de Factores Determinantes:**
   - Identificar las variables más influyentes en los niveles de ingreso
   - Comprender la relación entre educación, horas trabajadas y ingresos

4. **Soporte a Decisiones Estratégicas:**
   - Crear un modelo predictivo que permita simular escenarios
   - Evaluar el impacto potencial de políticas educativas o laborales

5. **Reducción del Índice de Pobreza:**
   - Generar insights accionables para reducir el % de población con ingresos bajos
   - Medir y monitorear la efectividad de las intervenciones

---

### **2. TIPO DE VARIABLES UTILIZADAS (01 punto)**

El dataset contiene **10 variables** clasificadas de la siguiente manera:

#### **A) Variables Numéricas Continuas (4 variables):**

| Variable | Tipo | Descripción | Rango |
|----------|------|-------------|-------|
| **EDAD** | Cuantitativa continua | Edad de la persona en años | 17 - 90 años |
| **CAPGANADO** | Cuantitativa continua | Ganancia de capital (valores, bonos, inmuebles) | 0 - máximo observado |
| **CAPPERD** | Cuantitativa continua | Pérdida de capital (valores, bonos, inmuebles) | 0 - máximo observado |
| **HORASEMANA** | Cuantitativa discreta | Horas trabajadas por semana (trabajo dependiente) | Variable |

#### **B) Variables Numéricas Discretas (1 variable):**

| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| **EDUCACIONNUM** | Cuantitativa discreta ordinal | Años dedicados a educación/estudio | Valores enteros |

#### **C) Variables Categóricas (4 variables):**

| Variable | Tipo | Descripción | Categorías Observadas |
|----------|------|-------------|----------------------|
| **EDUCACION** | Cualitativa nominal ordinal | Nivel educativo alcanzado | Bachelors, HS-grad, 11th, Masters, 9th, Some-college, Assoc-acdm, Assoc-voc, 7th-8th, etc. |
| **ESTADOCIV** | Cualitativa nominal | Estado civil de la persona | Nunca-casado, Casado-civil, Divorciado, Casado-conyuge-ausente, etc. |
| **SEXO** | Cualitativa nominal binaria | Género/sexo de la persona | Masculino, Femenino |
| **INGRESO** | Cualitativa nominal binaria **(VARIABLE OBJETIVO)** | Nivel de ingreso | <=50K, >50K |

#### **D) Variable Identificadora (1 variable):**

| Variable | Tipo | Descripción |
|----------|------|-------------|
| **CUSTOMER_ID** | Identificador único | ID del registro (formato: ID-00PP001, ID-00PP002, ...) |

#### **Análisis de Variables:**

**Variables Predictoras (Features):**
- **Numéricas:** EDAD, CAPGANADO, CAPPERD, HORASEMANA, EDUCACIONNUM (5 variables)
- **Categóricas:** EDUCACION, ESTADOCIV, SEXO (3 variables)
- **Total predictoras:** 8 variables

**Variable Objetivo (Target):**
- **INGRESO** (binaria: <=50K, >50K)

**Observaciones Importantes:**
1. **EDUCACION** y **EDUCACIONNUM** contienen información similar pero en diferentes formatos (categórica vs numérica)
2. **CAPGANADO** y **CAPPERD** pueden tener valores cero para personas sin transacciones de capital
3. **HORASEMANA** puede variar significativamente (desde part-time hasta más de 40 horas)
4. **SEXO** es binaria, lo que facilitará su codificación
5. **ESTADOCIV** tiene múltiples categorías que requerirán encoding

---

### **3. ANÁLISIS EXPLORATORIO DE DATOS (EDA) (02 puntos)**

#### **3.1 Dimensiones del Dataset**

```
Registros totales: 32,561 personas
Variables: 10 columnas
Valores nulos: 0 (dataset completo sin valores faltantes)
Memoria utilizada: 2.5+ MB
```

#### **3.2 Análisis de la Variable Objetivo: INGRESO**

**Distribución de clases:**
- **<=50K:** 24,720 registros (≈75.9%)
- **>50K:** 7,841 registros (≈24.1%)

**Observación crítica:** 
- Existe un **desbalance de clases** significativo (ratio 3:1)
- La mayoría de la población tiene ingresos ≤$50K
- Esto requerirá técnicas de balanceo para evitar sesgo en los modelos

#### **3.3 Análisis de Variables Numéricas**

| Variable | Media | Desv. Std | Mín | Q1 (25%) | Mediana (50%) | Q3 (75%) | Máx |
|----------|-------|-----------|-----|----------|---------------|----------|-----|
| **EDAD** | 38.58 | 13.64 | 17 | 28 | 37 | 48 | 90 |
| **CAPGANADO** | Variable | Variable | 0 | 0 | 0 | Variable | Variable |
| **CAPPERD** | Variable | Variable | 0 | 0 | 0 | Variable | Variable |
| **HORASEMANA** | Variable | Variable | Variable | Variable | Variable | Variable | Variable |
| **EDUCACIONNUM** | Variable | Variable | Variable | Variable | Variable | Variable | Variable |

**Insights de EDAD:**
- La edad media es **38.6 años** (población económicamente activa)
- La mediana es **37 años** (distribución relativamente simétrica)
- Rango amplio: desde **17 hasta 90 años**
- El 50% central de la población está entre **28 y 48 años**

**Insights de CAPGANADO y CAPPERD:**
- La mediana de ambas es **0**, lo que indica que la mayoría de las personas NO tienen ganancias/pérdidas de capital
- Estas variables tendrán **alta concentración en cero** (distribución asimétrica)
- Solo una minoría de personas tiene inversiones en valores, bonos o inmuebles

#### **3.4 Análisis de Variables Categóricas**

**SEXO:**
- **Masculino:** 21,790 registros (≈66.9%)
- **Femenino:** 10,771 registros (≈33.1%)
- Desbalance de género en la muestra

**EDUCACION (niveles observados en primeras 20 filas):**
- Bachelors (licenciatura)
- HS-grad (graduado de secundaria)
- Masters (maestría)
- Some-college (universidad incompleta)
- Assoc-acdm (asociado académico)
- Assoc-voc (asociado vocacional)
- 11th, 9th, 7th-8th (grados escolares incompletos)

**ESTADOCIV (estados observados en primeras 20 filas):**
- Casado-civil
- Nunca-casado
- Divorciado
- Casado-conyuge-ausente

#### **3.5 Patrones Iniciales Observados**

Del análisis de las primeras 20 filas, se observan patrones interesantes:

**Patrón 1 - Educación e Ingresos:**
- Personas con **Masters** tienden a tener **>50K** (ej: fila 8, 19)
- Personas con **Bachelors** pueden tener ambos niveles de ingreso
- Personas con **educación incompleta** (11th, 7th-8th) tienden a **<=50K**

**Patrón 2 - Horas Trabajadas:**
- Personas con **40+ horas/semana** y educación alta → más probabilidad de >50K
- Horas bajas (13, 16) correlacionan con <=50K

**Patrón 3 - Capital Ganado:**
- Personas con **CAPGANADO > 0** tienden a tener **>50K**
- Ejemplos: fila 8 (CAPGANADO=14,084), fila 9 (CAPGANADO=5,178) ambos >50K

**Patrón 4 - Género:**
- Hay representación de ambos géneros en ambas categorías de ingreso
- Requiere análisis más profundo para determinar correlación

**Patrón 5 - Estado Civil:**
- **Casado-civil** aparece frecuentemente en personas con **>50K**
- **Nunca-casado** y **Divorciado** son comunes en ambas categorías

#### **3.6 Análisis de Calidad de Datos**

**Fortalezas:**
- ✅ **Sin valores nulos** (32,561 registros completos)
- ✅ **Tipos de datos correctos** (int64 para numéricas, object para categóricas)
- ✅ **IDs únicos** (CUSTOMER_ID)
- ✅ **Rango de valores coherente** (edad 17-90, horas positivas)

**Consideraciones:**
- ⚠️ **Desbalance de clases** en INGRESO (75.9% vs 24.1%)
- ⚠️ **Desbalance de género** (66.9% Masculino vs 33.1% Femenino)
- ⚠️ **Asimetría en variables de capital** (mayoría con valor 0)
- ⚠️ **Redundancia potencial** entre EDUCACION y EDUCACIONNUM

---

### **4. DIVISIÓN DE DATOS: ENTRENAMIENTO Y VALIDACIÓN (01 punto)**

#### **4.1 Estrategia de División**

Para evaluar correctamente los modelos de Machine Learning, dividiremos el dataset en dos conjuntos:

**Propuesta de División:**

```
Dataset Total: 32,561 registros
│
├── Conjunto de ENTRENAMIENTO (Training Set): 70% = 22,793 registros
│   └── Utilizado para: Entrenar/ajustar los parámetros del modelo
│
└── Conjunto de VALIDACIÓN (Testing Set): 30% = 9,768 registros
    └── Utilizado para: Evaluar el rendimiento y generalización del modelo
```

#### **4.2 Justificación de la División 70/30**

**¿Por qué 70% entrenamiento y 30% validación?**

1. **Suficiente datos de entrenamiento:** 
   - Con 22,793 registros para entrenamiento, el modelo tendrá suficiente información para aprender patrones
   - Esto es especialmente importante con 8 variables predictoras

2. **Validación robusta:**
   - 9,768 registros de validación proporcionan una evaluación estadísticamente significativa
   - Permite calcular métricas confiables de rendimiento

3. **Balance adecuado:**
   - Para datasets de tamaño medio (30K-50K), 70/30 es estándar
   - Alternativa común: 80/20 (pero 70/30 da más datos para validación)

#### **4.3 Consideraciones Importantes**

**A) Estratificación por Variable Objetivo:**

Debido al desbalance de clases (75.9% <=50K vs 24.1% >50K), es **CRÍTICO** utilizar **muestreo estratificado**:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.30,           # 30% para validación
    random_state=42,          # Reproducibilidad
    stratify=y                # MANTIENE LA PROPORCIÓN DE CLASES
)
```

**Resultado esperado:**
- **Training Set:** 75.9% <=50K, 24.1% >50K
- **Testing Set:** 75.9% <=50K, 24.1% >50K

**B) Seed para Reproducibilidad:**
- Usar `random_state=42` (o cualquier número fijo)
- Garantiza que los resultados sean reproducibles

**C) No utilizar datos de validación durante entrenamiento:**
- Los datos de validación deben permanecer "ocultos" hasta la evaluación final
- Esto simula el rendimiento del modelo con datos nuevos

#### **4.4 Preparación de Variables**

**Antes de dividir los datos, realizar:**

1. **Separar variable objetivo:**
   ```
   X = todas las variables predictoras (EDAD, CAPGANADO, ..., SEXO)
   y = variable objetivo (INGRESO)
   ```

2. **Eliminar variables no predictoras:**
   - Remover **CUSTOMER_ID** (identificador, no aporta información predictiva)

3. **Codificación de variables categóricas** (pendiente para pregunta 5):
   - EDUCACION → One-Hot Encoding
   - ESTADOCIV → One-Hot Encoding
   - SEXO → Label Encoding (0/1)
   - INGRESO (target) → Label Encoding (0=<=50K, 1=>50K)

#### **4.5 Alternativa: Validación Cruzada (Cross-Validation)**

Para modelos más robustos, se podría considerar **K-Fold Cross-Validation** (K=5 o K=10):

```
Dataset Total: 32,561 registros
│
└── Dividir en K=5 folds
    ├── Fold 1: Validación | Fold 2-5: Entrenamiento
    ├── Fold 2: Validación | Fold 1,3-5: Entrenamiento
    ├── Fold 3: Validación | Fold 1-2,4-5: Entrenamiento
    ├── Fold 4: Validación | Fold 1-3,5: Entrenamiento
    └── Fold 5: Validación | Fold 1-4: Entrenamiento
    
    Métrica final = Promedio de las 5 iteraciones
```

**Ventajas:**
- Utiliza todos los datos tanto para entrenamiento como validación
- Reduce varianza en la estimación de rendimiento
- Detecta overfitting más efectivamente

**Desventaja:**
- Requiere entrenar el modelo K veces (más costoso computacionalmente)

#### **4.6 Métricas de Evaluación a Utilizar (Pregunta 6)**

Dado el **desbalance de clases**, las métricas a evaluar serán:

1. **Accuracy (Exactitud):** % de predicciones correctas
2. **Precision:** De los que predecimos >50K, ¿cuántos realmente son >50K?
3. **Recall (Sensibilidad):** De los que realmente son >50K, ¿cuántos detectamos?
4. **F1-Score:** Media armónica de Precision y Recall
5. **Matriz de Confusión:** Visualización de VP, VN, FP, FN
6. **ROC-AUC:** Área bajo la curva ROC
7. **Control de calidad al 5%:** Intervalo de confianza del 95% para las predicciones

## 📊 CARACTERÍSTICAS DEL DATASET - RESUMEN

| Característica | Detalle |
|---------------|---------|
| **Total de registros** | 32,561 personas |
| **Variables numéricas** | 5 (EDAD, CAPGANADO, CAPPERD, HORASEMANA, EDUCACIONNUM) |
| **Variables categóricas** | 4 (EDUCACION, ESTADOCIV, SEXO, INGRESO) |
| **Variable identificadora** | 1 (CUSTOMER_ID) |
| **Valores nulos** | 0 (dataset completo) |
| **Desbalance de clases** | Sí (75.9% <=50K, 24.1% >50K) |
| **Tipo de problema** | Clasificación binaria (aunque se solicite regresión) |
| **Metodología** | CRISP-DM |
| **División propuesta** | 70% entrenamiento, 30% validación |

---

## 🚀 CONCLUSIONES DEL ANÁLISIS PRELIMINAR

1. **Dataset robusto:** 32,561 registros completos sin valores nulos
2. **Desbalance de clases:** Requiere estratificación y posiblemente técnicas de balanceo (SMOTE, class_weight)
3. **Variables relevantes:** Educación, horas trabajadas y capital ganado parecen ser predictores fuertes
4. **Redundancia:** EDUCACION y EDUCACIONNUM podrían generar multicolinealidad
5. **Preparación necesaria:** Encoding de categóricas, escalamiento de numéricas, manejo de outliers
6. **Interpretabilidad:** Los coeficientes del modelo permitirán identificar factores clave de pobreza
7. **Aplicación práctica:** Los resultados servirán para focalizar políticas públicas en grupos vulnerables

---

## 📁 ESTRUCTURA DEL PROYECTO

```
TrabajoIndividualN2/
│
├── CasoEstudio05_Censo.pdf                    # Documento con el caso de estudio
├── CensoPoblacion.xlsx                        # Dataset original (32,561 registros)
├── README.md                                  # Este archivo (análisis y respuestas 1-4)
├── RESUMEN_EJECUTIVO.md                       # Guía rápida del proyecto
├── GUIA_IMPLEMENTACION.md                     # ✨ NUEVO: Código completo para cada notebook
│
├── notebooks/                                 # ✅ CREADOS
│   ├── 01_EDA_Completo.ipynb                 # ✅ Análisis exploratorio (Pregunta 3)
│   ├── 02_Preprocesamiento.ipynb             # ✅ Preparación de datos (Pregunta 4)
│   ├── 03_Modelo1_RegresionLogistica.ipynb   # ✅ Modelo 1 (Pregunta 5)
│   ├── 04_Modelo2_RegresionLineal.ipynb      # ✅ Modelo 2 (Pregunta 5)
│   └── 05_Comparacion_Validacion_Final.ipynb # ✅ Validación (Pregunta 6)
│
└── resultados/                                # Se creará al ejecutar notebooks
    ├── distribucion_*.png                     # Gráficos del EDA
    ├── X_train.csv, X_test.csv               # Datos preprocesados
    ├── y_train.csv, y_test.csv               # Labels
    ├── modelo_logistica.pkl                   # Modelo 1 entrenado
    ├── modelo_lineal.pkl                      # Modelo 2 entrenado
    ├── comparacion_modelos.csv                # Tabla comparativa
    ├── matrices_confusion.png                 # Visualizaciones
    └── curvas_roc.png                        # Curvas ROC
```
