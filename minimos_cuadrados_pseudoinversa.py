#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from math import isnan

# ------------- Config -------------
INPUT_CSV = "csvbueno.csv"

# ----------------------------
# 1) Leer CSV probando encodings
# ----------------------------
df = None
last_exc = None
for enc in ('utf-8', 'latin-1', 'cp1252', 'ascii'):
    try:
        # algunos csv usan ; como separador; si no, pandas detectará
        df = pd.read_csv(INPUT_CSV, sep=';', engine='python', encoding=enc, skipinitialspace=True)
        print(f"Leído {INPUT_CSV} con encoding={enc!r} y sep=';'")
        break
    except Exception as e:
        last_exc = e
# si falló con ; probamos sin sep
if df is None:
    for enc in ('utf-8', 'latin-1', 'cp1252', 'ascii'):
        try:
            df = pd.read_csv(INPUT_CSV, encoding=enc, skipinitialspace=True)
            print(f"Leído {INPUT_CSV} con encoding={enc!r} y sep=',' (fallback)")
            break
        except Exception as e:
            last_exc = e

if df is None:
    raise last_exc

print(f"DataFrame shape: {df.shape}")
print("Columnas:", df.columns.tolist())

# ------------------------------------------------
# 2) Funciones de extracción / conversión
# ------------------------------------------------
def to_numeric_general(value):
    """Extrae números; si hay varios (rango), devuelve el promedio."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    # buscamos todos los números (enteros o decimales) y promediamos
    nums = re.findall(r"[-+]?\d*[\.,]?\d+", s)
    if not nums:
        return np.nan
    try:
        nums_f = [float(x.replace(",", ".")) for x in nums]
        return float(sum(nums_f) / len(nums_f))
    except:
        return np.nan

def correct_hp_column(valor):
    """Extrae un número representativo de HorsePower (promedio si hay rango)."""
    return to_numeric_general(valor)

def extract_cc(valor):
    """
    Extrae el primer número encontrado en la cadena `valor`.
    Acepta formatos con comas como "1,200 cc" o rangos "1,000 - 2,000 cc" o "1798 / 1987 cc + batt".
    Devuelve float (sin separadores) o np.nan si no encuentra números.
    """
    if valor is None:
        return np.nan
    s = str(valor)
    # buscar números con posible separador de miles y decimales
    m = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+)(?=[^\d]|$)", s)
    if not m:
        return np.nan
    num = m.group(0).replace(',', '')
    try:
        return float(num)
    except Exception:
        return np.nan

def extract_torque(valor):
    """Extrae número representativo de Torque (promedia rangos)."""
    return to_numeric_general(valor)

# ----------------------------
# 3) Aplicar limpieza a columnas
# ----------------------------
# Columnas esperadas
col_hp = "HorsePower"
col_cc = "CC/Battery Capacity"
col_torque = "Torque"
col_ts = "Total Speed"

for c in (col_hp, col_cc, col_torque, col_ts):
    if c not in df.columns:
        raise ValueError(f"La columna esperada '{c}' no se encuentra en el CSV. Columnas disponibles: {df.columns.tolist()}")

# Diagnóstico previo
print("Valores no nulos antes de limpieza:", df[[col_hp, col_cc, col_torque, col_ts]].notna().sum().to_dict())

# Creamos mask_hp: filas inicialmente con HorsePower y Total Speed no nulos (para comparar y mantener Y)
mask_hp = df[col_hp].notna() & df[col_ts].notna()
n_hp = int(mask_hp.sum())
print(f"Filas con HorsePower y Total Speed presentes (mask_hp): {n_hp}")

# Aplicar conversiones
df[col_hp] = df[col_hp].apply(correct_hp_column)
df[col_cc] = df[col_cc].apply(extract_cc)
df[col_torque] = df[col_torque].apply(extract_torque)
# Total Speed: extraer número (p. ej. "250 km/h" -> 250)
df[col_ts] = df[col_ts].astype(str).str.extract(r"([-+]?\d*[\.,]?\d+)")[0].replace("", np.nan).map(lambda v: float(str(v).replace(",", ".")) if pd.notna(v) else np.nan)

# Ver conteos después de extracción
print("Valores no nulos después de extracción:", df[[col_hp, col_cc, col_torque, col_ts]].notna().sum().to_dict())

# ----------------------------
# 4) Imputación por mediana para CC y Torque entre las filas usadas por mask_hp (conservar n_hp)
# ----------------------------
cc_in_mask = df.loc[mask_hp, col_cc]
torque_in_mask = df.loc[mask_hp, col_torque]

cc_nonnull = int(cc_in_mask.notna().sum())
torque_nonnull = int(torque_in_mask.notna().sum())
cc_missing = n_hp - cc_nonnull
torque_missing = n_hp - torque_nonnull

#print(f"Entre las {n_hp} filas originales: CC numérico = {cc_nonnull}, faltantes = {cc_missing}")
#print(f"Entre las {n_hp} filas originales: Torque numérico = {torque_nonnull}, faltantes = {torque_missing}")

# imputar mediana calculada sobre las filas mask_hp que tengan valor
median_cc = cc_in_mask.median()
median_torque = torque_in_mask.median()
#print(f"Mediana CC (mask_hp): {median_cc}, Mediana Torque (mask_hp): {median_torque}")

# Rellenar en el DataFrame SOLO en las filas que están en mask_hp
df.loc[mask_hp, col_cc] = df.loc[mask_hp, col_cc].fillna(median_cc)
df.loc[mask_hp, col_torque] = df.loc[mask_hp, col_torque].fillna(median_torque)

# Ahora tomar solo las filas que pertenecen a mask_hp y que además tengan Total Speed no nulo y HorsePower no nulo
df_model = df.loc[mask_hp, [col_hp, col_cc, col_torque, col_ts]].dropna().reset_index(drop=True)
#print(f"Filas disponibles para el modelo (después de imputación y dropna final): {df_model.shape[0]}")

if df_model.shape[0] < 3:
    raise ValueError("Pocos datos para ajustar regresión múltiple (menos de 3 filas)")

# ----------------------------
# 5) Preparar X, y (ya no necesitamos normalizar)
# ----------------------------
X = df_model[[col_hp, col_cc, col_torque]].values.astype(float)  # orden x1,x2,x3
y = df_model[col_ts].values.astype(float)

# Agregar columna de 1s para intercepto
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# ----------------------------
# 6) Ajuste por mínimos cuadrados usando pseudoinversa
# ----------------------------
theta = np.linalg.pinv(X_b).dot(y)

# Extraer coeficientes
b0, b1, b2, b3 = theta

# Predicciones y R2
y_pred = X_b.dot(theta)
y_mean = y.mean()
SST = np.sum((y - y_mean)**2)
SSR = np.sum((y_pred - y_mean)**2)
SSE = np.sum((y - y_pred)**2)
R2_model = SSR / SST if SST != 0 else np.nan

print("\n--- RESULTADOS REGRESIÓN MÚLTIPLE (Mínimos Cuadrados / Pseudoinversa) ---")
print(f"n = {len(y)}")
print(f"Coeficientes (unidades reales):")
print(f"b0 = {b0:.6f}")
print(f"b1 (x1 HorsePower) = {b1:.6f}")
print(f"b2 (x2 CC) = {b2:.6f}")
print(f"b3 (x3 Torque) = {b3:.6f}")
print(f"Ecuación: Y = {b0:.4f} + {b1:.4f}*x1 + {b2:.4f}*x2 + {b3:.4f}*x3")
print(f"R² = {R2_model:.6f}")
print(f"SSE = {SSE:.4f}, SSR = {SSR:.4f}, SST = {SST:.4f}")

# ----------------------------
# 8) imprimir la matriz de correlación para reporte
# ----------------------------
print("\nMatriz de correlación (variables usadas):")
print(df_model[[col_hp, col_cc, col_torque, col_ts]].corr().to_string())