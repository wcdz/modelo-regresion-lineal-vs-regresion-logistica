# 📋 RESUMEN EJECUTIVO - Análisis Rápido

## 🎯 TU TAREA EN RESUMEN

Tienes que **predecir si una persona gana <=50K o >50K** usando datos de censo poblacional.

---

## 📊 DATOS QUE TIENES

- **32,561 personas** en el dataset
- **8 variables predictoras:** edad, educación, horas trabajadas, capital ganado/perdido, estado civil, sexo
- **1 variable objetivo:** INGRESO (<=50K o >50K)
- **Desbalance:** 75.9% gana <=50K, solo 24.1% gana >50K

---

## ✅ PREGUNTAS 1-4 (YA RESPONDIDAS EN README.md)

### Pregunta 1: Definición del Problema
- **Tipo:** Clasificación binaria (aunque pide regresión lineal)
- **Objetivo:** Identificar poblaciones vulnerables para políticas públicas
- **Aplicación:** Reducir % de pobreza focalizando recursos

### Pregunta 2: Tipos de Variables
- **5 numéricas:** EDAD, CAPGANADO, CAPPERD, HORASEMANA, EDUCACIONNUM
- **3 categóricas:** EDUCACION, ESTADOCIV, SEXO
- **1 objetivo:** INGRESO (<=50K, >50K)

### Pregunta 3: Análisis Exploratorio (EDA)
- **Edad promedio:** 38.6 años
- **Sin valores nulos:** Dataset completo
- **Patrones encontrados:**
  - Más educación → más ingreso
  - Más horas trabajadas → más ingreso
  - Capital ganado > 0 → generalmente >50K
  - Mayoría de personas sin ganancias/pérdidas de capital

### Pregunta 4: División de Datos
- **70% entrenamiento** (22,793 registros)
- **30% validación** (9,768 registros)
- **Estratificación:** Mantener proporción 75.9% / 24.1% en ambos conjuntos

---

## 🔥 PREGUNTAS 5-6 (PARA HACER EN CÓDIGO)

### Pregunta 5: Construir 2 Modelos (10 puntos)

**IMPORTANTE:** El PDF dice "regresión lineal" pero la variable objetivo es **categórica**, así que:

#### **MODELO 1: Regresión Logística** ⭐ (RECOMENDADO)
- Es técnicamente regresión pero para clasificación
- Predice probabilidad de >50K
- Coeficientes interpretables

#### **MODELO 2: Regresión Lineal Múltiple**
- Codificar INGRESO como 0 (<=50K) y 1 (>50K)
- Aplicar umbral 0.5 para clasificar
- Menos adecuado pero cumple con "regresión lineal"

**Tareas:**
- Preprocesar datos (codificar categóricas, escalar numéricas)
- Entrenar ambos modelos con datos de entrenamiento (70%)
- Comparar modelos
- Interpretar coeficientes (¿qué variables influyen más?)

### Pregunta 6: Validación (4 puntos)

**Tareas:**
- Evaluar modelos con datos de validación (30%)
- Calcular métricas:
  - **Accuracy** (exactitud general)
  - **Precision** (de los que predigo >50K, cuántos acierto)
  - **Recall** (de los >50K reales, cuántos detecto)
  - **F1-Score** (balance entre precision y recall)
  - **Matriz de confusión**
- Calcular **% de acierto con IC del 95%** (control de calidad 5%)
- Dar **conclusiones** y **recomendaciones**

---

## 🛠️ PASOS PARA CODEAR (NOTEBOOK)

### Paso 1: Importar Librerías
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
```

### Paso 2: Cargar Datos
```python
df = pd.read_excel('CensoPoblacion.xlsx', sheet_name='adult')
```

### Paso 3: EDA Visual (Pregunta 3 extendida)
- Histogramas de variables numéricas
- Boxplots para detectar outliers
- Gráficos de barras para categóricas
- Correlación entre variables

### Paso 4: Preprocesamiento
- Eliminar CUSTOMER_ID
- Codificar INGRESO: <=50K=0, >50K=1
- Codificar SEXO: Masculino=1, Femenino=0
- One-Hot Encoding para EDUCACION y ESTADOCIV
- Escalar variables numéricas (StandardScaler)

### Paso 5: Dividir Datos (Pregunta 4)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
```

### Paso 6: Modelo 1 - Regresión Logística (Pregunta 5)
```python
modelo1 = LogisticRegression(max_iter=1000, random_state=42)
modelo1.fit(X_train, y_train)
y_pred1 = modelo1.predict(X_test)
```

### Paso 7: Modelo 2 - Regresión Lineal (Pregunta 5)
```python
modelo2 = LinearRegression()
modelo2.fit(X_train, y_train)
y_pred2 = (modelo2.predict(X_test) >= 0.5).astype(int)
```

### Paso 8: Interpretar Coeficientes (Pregunta 5)
```python
coeficientes = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': modelo1.coef_[0]
}).sort_values('Coeficiente', ascending=False)
```

### Paso 9: Evaluar Modelos (Pregunta 6)
```python
# Métricas Modelo 1
acc1 = accuracy_score(y_test, y_pred1)
precision1 = precision_score(y_test, y_pred1)
recall1 = recall_score(y_test, y_pred1)
f1_1 = f1_score(y_test, y_pred1)

# Matriz de confusión
cm1 = confusion_matrix(y_test, y_pred1)

# Intervalo de confianza 95%
from scipy import stats
n = len(y_test)
error_std = np.sqrt(acc1 * (1 - acc1) / n)
ic_95 = stats.norm.interval(0.95, loc=acc1, scale=error_std)
```

### Paso 10: Conclusiones (Pregunta 6)
- ¿Qué modelo es mejor?
- ¿Qué variables influyen más en los ingresos?
- ¿Qué grupos son más vulnerables?
- Recomendaciones para políticas públicas

---

## 🎯 VARIABLES QUE MÁS INFLUYEN (HIPÓTESIS)

Basado en el análisis preliminar, esperamos que estas variables sean **más importantes**:

1. **EDUCACIONNUM** / **EDUCACION** ⭐⭐⭐
   - Más años de educación → más ingreso
   - Masters/Bachelors → >50K

2. **CAPGANADO** ⭐⭐⭐
   - Personas con inversiones tienden a ganar más

3. **HORASEMANA** ⭐⭐
   - Más horas trabajadas → más ingreso

4. **EDAD** ⭐⭐
   - Relación no lineal (aumenta hasta ~50 años, luego disminuye)

5. **ESTADOCIV** ⭐
   - Casados tienden a ganar más

6. **SEXO** ⭐
   - Posible brecha salarial de género

7. **CAPPERD** ⭐
   - Menos influyente

---

## ⚠️ PROBLEMAS A CONSIDERAR

### 1. **Desbalance de Clases (75.9% vs 24.1%)**
**Soluciones:**
- Usar `stratify=y` en train_test_split
- Usar `class_weight='balanced'` en LogisticRegression
- Aplicar SMOTE para balancear clases
- No usar solo Accuracy, usar F1-Score

### 2. **Redundancia: EDUCACION vs EDUCACIONNUM**
**Solución:**
- Usar solo una de las dos
- O verificar multicolinealidad con VIF

### 3. **Asimetría en CAPGANADO y CAPPERD**
**Solución:**
- Aplicar transformación logarítmica: `log(x + 1)`
- O crear variable binaria: tiene_capital (0/1)

### 4. **Outliers**
**Solución:**
- Detectar con boxplots
- Aplicar winsorization o eliminar si es necesario

### 5. **Escalamiento**
**Solución:**
- StandardScaler para todas las numéricas
- Especialmente importante para regresión lineal

---

## 📊 MÉTRICAS ESPERADAS

Con un buen modelo, deberías obtener:

- **Accuracy:** 80-85%
- **Precision:** 70-75% (para clase >50K)
- **Recall:** 60-70% (para clase >50K)
- **F1-Score:** 65-72%

Si obtienes **Accuracy > 90%**, verifica si hay **data leakage** o sobreajuste.

---

## 🚀 SIGUIENTES PASOS INMEDIATOS

1. **Crear un Jupyter Notebook** (recomendado dividir en 2):
   - `Notebook_Preguntas_5.ipynb` → Modelos
   - `Notebook_Preguntas_6.ipynb` → Validación

2. **Implementar el código paso a paso** (ver sección "Pasos para Codear")

3. **Comparar ambos modelos** con tabla de métricas:

| Métrica | Modelo 1 (Logística) | Modelo 2 (Lineal) |
|---------|---------------------|-------------------|
| Accuracy | ? | ? |
| Precision | ? | ? |
| Recall | ? | ? |
| F1-Score | ? | ? |

4. **Interpretar coeficientes:**
   - ¿EDUCACIONNUM tiene coeficiente positivo? ✅
   - ¿HORASEMANA tiene coeficiente positivo? ✅
   - ¿Qué variable tiene mayor impacto?

5. **Conclusiones para políticas públicas:**
   - Invertir en educación
   - Promover empleo de tiempo completo
   - Focalizar en personas con baja educación y pocas horas trabajadas

---

## 📝 ESTRUCTURA FINAL DE ENTREGA

```
1. README.md (ya creado) ✅
   - Respuestas teóricas preguntas 1-4

2. Notebook(s) con código (a crear) ⏳
   - Pregunta 5: Construcción de 2 modelos
   - Pregunta 6: Validación y conclusiones

3. Resultados y visualizaciones
   - Matriz de confusión
   - Gráficos de coeficientes
   - Tabla comparativa de modelos

4. Conclusiones y recomendaciones
   - Mejor modelo
   - Variables más importantes
   - Recomendaciones para ONG/gobierno
```

---

## ✅ CHECKLIST RÁPIDO

**Antes de empezar a programar:**
- [x] Leer PDF completo
- [x] Explorar Excel
- [x] Entender el problema (clasificación binaria)
- [x] Responder preguntas 1-4 teóricas

**Al programar (Pregunta 5):**
- [ ] Cargar datos
- [ ] EDA visual completo
- [ ] Preprocesar (encoding, scaling)
- [ ] Dividir train/test (70/30 estratificado)
- [ ] Entrenar Modelo 1 (Logística)
- [ ] Entrenar Modelo 2 (Lineal)
- [ ] Interpretar coeficientes
- [ ] Comparar modelos

**Al validar (Pregunta 6):**
- [ ] Predecir con datos de test
- [ ] Calcular Accuracy, Precision, Recall, F1
- [ ] Matriz de confusión
- [ ] IC del 95% para % de acierto
- [ ] Conclusiones
- [ ] Recomendaciones para negocio

---

**¿Listo para programar? Crea el notebook y empieza con el Paso 1! 🚀**

