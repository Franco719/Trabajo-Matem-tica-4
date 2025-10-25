import pandas as pd
import re
import numpy as np

df = None
last_exc = None
for enc in ('utf-8', 'latin-1', 'cp1252', 'ascii'):
    try:
        df = pd.read_csv('csvbueno.csv', sep=';', engine='python', encoding=enc, skipinitialspace=True)
        print(f"Leído csvbueno.csv con encoding={enc!r} y sep=';'")
        break
    except Exception as e:
        last_exc = e
if df is None:
    raise last_exc

# diagnóstico rápido tras la lectura
print(f"Leído DataFrame con shape={df.shape}")
print("Columnas:", df.columns.tolist())
if 'HorsePower' in df.columns and 'Total Speed' in df.columns:
    print("Non-null counts (HorsePower, Total Speed):",
          df[['HorsePower', 'Total Speed']].notna().sum().to_dict())
    print("Primeras filas:")
    print(df.head(5))



def correct_hp_column(valor):
    if isinstance(valor, str):
        # extrae números (enteros o decimales) de cadenas tipo "130 hp", "120.5", "NA", etc.
        m = re.search(r'[-+]?\d+(\.\d+)?', valor.replace(',', '.'))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return np.nan
        return np.nan
    try:
        return float(valor)
    except Exception:
        return np.nan

# Limpieza y conversión de las columnas de interés
x_col = 'HorsePower'
y_col = 'Total Speed'

# contar antes de la limpieza
before_hp_nonnull = df[x_col].notna().sum() if x_col in df.columns else 0
before_ts_nonnull = df[y_col].notna().sum() if y_col in df.columns else 0
print(f"Antes limpieza: HorsePower non-null={before_hp_nonnull}, Total Speed non-null={before_ts_nonnull}")

# máscara de filas usadas originalmente por HorsePower (para mantener la misma Y)
mask_hp = (df[x_col].notna()) & (df[y_col].notna())
n_hp = int(mask_hp.sum())
print(f"Filas usadas originalmente (mask_hp) n = {n_hp}")

# limpiar HorsePower
df[x_col] = df[x_col].apply(correct_hp_column)

# convertir Total Speed extrayendo la parte numérica (p. ej. '250 km/h' -> 250)
if y_col in df.columns:
    df[y_col] = pd.to_numeric(df[y_col].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors='coerce')
else:
    df[y_col] = np.nan

# contar después de la limpieza
after_hp_nonnull = df[x_col].notna().sum()
after_ts_nonnull = df[y_col].notna().sum()
print(f"Después limpieza: HorsePower non-null={after_hp_nonnull}, Total Speed non-null={after_ts_nonnull}")

# eliminar filas con NA en X o Y
data = df[[x_col, y_col]].dropna().reset_index(drop=True)
X = data[x_col].values.astype(float)
Y = data[y_col].values.astype(float)
n = len(X)

if n < 2:
    raise ValueError("No hay suficientes datos válidos para ajustar una regresión lineal (n < 2).")

# medias
x_bar = X.mean()
y_bar = Y.mean()

# coeficiente B1 (pendiente) y B0 (intercepto)
num = np.sum((X - x_bar) * (Y - y_bar))
den = np.sum((X - x_bar) ** 2)
B1 = num / den
B0 = y_bar - B1 * x_bar

# Sxy y Sxx (sumas centradas) para transparencia y verificación
Sxy = num   # suma de (xi - x_bar)(yi - y_bar)
Sxx = den   # suma de (xi - x_bar)^2

print(f"Sxy = {Sxy:.4f}")
print(f"Sxx = {Sxx:.4f}")
print(f"B1 calculado como Sxy/Sxx = {Sxy/Sxx:.6f}")

# predicciones y estadísticas de ajuste
Y_pred = B0 + B1 * X
residuals = Y - Y_pred
SSE = np.sum(residuals ** 2)            # suma residuos al cuadrado
SSR = np.sum((Y_pred - y_bar) ** 2)     # regresión al cuadrado (también = b1 * Sxy)
SST = np.sum((Y - y_bar) ** 2)          # total (Syy)
Syy = SST

# Verificaciones por identidad
SSR_via_identity = (Sxy ** 2) / Sxx if Sxx != 0 else np.nan
SSE_via_identity = Syy - SSR_via_identity if not np.isnan(SSR_via_identity) else np.nan

R2 = SSR / Syy if Syy != 0 else np.nan

print(f"Syy = {Syy:.4f}")
print(f"SSR (directo) = {SSR:.4f}")
print(f"SSR (via Sxy^2/Sxx) = {SSR_via_identity:.4f}")
print(f"SSE (directo) = {SSE:.4f}")
print(f"SSE (via Syy-SSR) = {SSE_via_identity:.4f}")
print(f"R² = {R2:.4f}")

# error estándar de la estimación y de los coeficientes
sigma2 = SSE / (n - 2)                  # varianza residual
se_B1 = np.sqrt(sigma2 / den)
se_B0 = np.sqrt(sigma2 * (1.0/n + x_bar**2 / den))

# impresión de resultados principales
print(f"n = {n}")
print(f"x̄ = {x_bar:.4f}")
print(f"ȳ = {y_bar:.4f}")
print(f"B1 (pendiente) = {B1:.6f}")
print(f"B0 (intercepto) = {B0:.6f}")
print(f"R² = {R2:.4f}")
print(f"SSE = {SSE:.4f}")
print(f"Error estándar pendiente (se_B1) = {se_B1:.6f}")
print(f"Error estándar intercepto (se_B0) = {se_B0:.6f}")

# función auxiliar para predecir
def predict(hp):
    try:
        hp_val = float(correct_hp_column(hp))
    except Exception:
        return np.nan
    return B0 + B1 * hp_val

# ejemplo: predecir para 150 HP si se quiere
# print("Predicción Total Speed para 150 HP:", predict(150))
# (Los gráficos se generan después del procesamiento de CC para asegurar que las variables existan.)

# ----------------------
# Nueva predictora: extraer CC/Battery Capacity
# ----------------------
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

# aplicar extracción a la columna
cc_col = 'CC/Battery Capacity'
if cc_col in df.columns:
    cc_values = df[cc_col].astype(str).apply(extract_cc)
else:
    cc_values = pd.Series([np.nan] * len(df), index=df.index)

# Contar cuántos CC tenemos en el conjunto original de filas usadas por HorsePower
cc_in_mask = cc_values[mask_hp]
cc_nonnull = int(cc_in_mask.notna().sum())
cc_missing = n_hp - cc_nonnull
print(f"Entre las {n_hp} filas originales, CC con valor numérico: {cc_nonnull}, faltantes: {cc_missing}")

# Función auxiliar para ajustar regresión dado X y Y (ya centrados en Y_original)
def fit_and_report(X_array, Y_array, label='predictor'):
    X = np.asarray(X_array, dtype=float)
    Y = np.asarray(Y_array, dtype=float)
    n_local = len(X)
    x_bar_local = X.mean()
    y_bar_local = Y.mean()
    Sxy_local = np.sum((X - x_bar_local) * (Y - y_bar_local))
    Sxx_local = np.sum((X - x_bar_local) ** 2)
    b1_local = Sxy_local / Sxx_local
    b0_local = y_bar_local - b1_local * x_bar_local
    Y_pred_local = b0_local + b1_local * X
    residuals_local = Y - Y_pred_local
    SSE_local = np.sum(residuals_local ** 2)
    SSR_local = np.sum((Y_pred_local - y_bar_local) ** 2)
    Syy_local = np.sum((Y - y_bar_local) ** 2)
    R2_local = SSR_local / Syy_local if Syy_local != 0 else np.nan
    print('\n--- Regresión usando', label, '---')
    print(f"n = {n_local}")
    print(f"b1 = {b1_local:.6f}, b0 = {b0_local:.6f}")
    print(f"Sxy = {Sxy_local:.4f}, Sxx = {Sxx_local:.4f}, Syy = {Syy_local:.4f}")
    print(f"SSR = {SSR_local:.4f}, SSE = {SSE_local:.4f}, R2 = {R2_local:.4f}")
    return dict(n=n_local, b1=b1_local, b0=b0_local, Sxy=Sxy_local, Sxx=Sxx_local, Syy=Syy_local, SSR=SSR_local, SSE=SSE_local, R2=R2_local)

# Si no hay faltantes, ajustamos directamente con las mismas filas (mismo n)
Y_original = df.loc[mask_hp, y_col].astype(float)
if cc_missing == 0:
    X_cc = cc_in_mask.astype(float)
    report_cc_same_n = fit_and_report(X_cc.values, Y_original.values, label='CC (first number) — same n')
else:
    # Ajuste con filas válidas donde CC existe
    X_cc_valid = cc_in_mask.dropna().astype(float)
    Y_cc_valid = df.loc[cc_in_mask.dropna().index, y_col].astype(float)
    report_valid = fit_and_report(X_cc_valid.values, Y_cc_valid.values, label='CC (only rows with CC)')
    # imprimir media de CC en filas válidas
    x_bar_cc_valid = X_cc_valid.mean() if len(X_cc_valid) > 0 else np.nan
    print(f"x̄ CC (filas válidas) = {x_bar_cc_valid:.4f}")

    # Ajuste con imputación por mediana para conservar n
    median_cc = cc_in_mask.median()
    print(f"Mediana CC (para imputación): {median_cc}")
    X_cc_imputed = cc_in_mask.fillna(median_cc).astype(float)
    report_imputed = fit_and_report(X_cc_imputed.values, Y_original.values, label='CC (imputación por mediana, conserva n)')
    # imprimir media de CC después de la imputación
    x_bar_cc_imputed = X_cc_imputed.mean()
    print(f"Mediana CC (para imputación): {median_cc}")
    print(f"x̄ CC (después imputación) = {x_bar_cc_imputed:.4f}")

    print('\nAdvertencia: hay', cc_missing, 'filas sin número extraído en CC entre las filas originales.\n'
        'Se mostró la regresión sólo con filas válidas y otra con imputación por mediana (conservando n).\n'
        'Decime si preferís otra estrategia (ej. imputación por media, por marca, o eliminar filas).')

# ----------------------
# Nueva predictora: Torque
# ----------------------
def extract_torque(valor):
    """
    Extrae el primer número de la columna Torque.
    Acepta formatos como '900 Nm', '100 - 140 Nm', '560 Nm' o '100 - 140 Nm'.
    Devuelve float o np.nan.
    """
    if valor is None:
        return np.nan
    s = str(valor)
    # Buscar el primer número (soporta separador de miles con comas)
    m = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+)(?=[^\d]|$)", s)
    if not m:
        return np.nan
    num = m.group(0).replace(',', '')
    try:
        return float(num)
    except Exception:
        return np.nan

# aplicar extracción a la columna Torque si existe
torque_col = 'Torque'
if torque_col in df.columns:
    torque_values = df[torque_col].astype(str).apply(extract_torque)
else:
    torque_values = pd.Series([np.nan] * len(df), index=df.index)

# Contar cuántos Torque tenemos en las filas usadas por HorsePower
torque_in_mask = torque_values[mask_hp]
torque_nonnull = int(torque_in_mask.notna().sum())
torque_missing = n_hp - torque_nonnull
print(f"Entre las {n_hp} filas originales, Torque con valor numérico: {torque_nonnull}, faltantes: {torque_missing}")

# Ajustes para Torque: válido y con imputación por mediana para conservar n
Y_for_torque = Y_original  # ya es la misma Y usada para comparar
if torque_missing == 0:
    X_torque = torque_in_mask.astype(float)
    report_torque_same_n = fit_and_report(X_torque.values, Y_for_torque.values, label='Torque (first number) — same n')
else:
    # A) Sólo filas válidas
    X_torque_valid = torque_in_mask.dropna().astype(float)
    Y_torque_valid = df.loc[torque_in_mask.dropna().index, y_col].astype(float)
    report_torque_valid = fit_and_report(X_torque_valid.values, Y_torque_valid.values, label='Torque (only rows with Torque)')
    x_bar_torque_valid = X_torque_valid.mean() if len(X_torque_valid) > 0 else np.nan
    print(f"x̄ Torque (filas válidas) = {x_bar_torque_valid:.4f}")

    # Imputación por mediana para conservar n
    median_torque = torque_in_mask.median()
    print(f"Mediana Torque (para imputación): {median_torque}")
    X_torque_imputed = torque_in_mask.fillna(median_torque).astype(float)
    report_torque_imputed = fit_and_report(X_torque_imputed.values, Y_for_torque.values, label='Torque (imputación por mediana, conserva n)')
    x_bar_torque_imputed = X_torque_imputed.mean()
    print(f"x̄ Torque (después imputación) = {x_bar_torque_imputed:.4f}")

    print('\nAdvertencia: hay', torque_missing, 'filas sin número extraído en Torque entre las filas originales.')