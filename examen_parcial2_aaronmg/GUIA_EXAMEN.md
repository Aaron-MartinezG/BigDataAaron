# Guía paso a paso — Examen B (K-Means, viajes en efectivo NYC)

Notebook: `notebooks/Examen_B_Martínez_Aarón.ipynb` · Datos: `data/raw/nyc_taxi_examen.csv`
Basada en el mismo flujo de `U4_clustering` (log → StandardScaler → KMeans → silueta → resumen por grupo).

---

## ⚠️ Lo que revisé en los datos (léelo primero)

| Hallazgo | Consecuencia |
|---|---|
| `payment_type` es **numérico** (1.0, 2.0, 3.0, 4.0, NaN), **no existe el texto `"Cash"`** | Si haces `df[df.payment_type == "Cash"]` te queda **vacío**. Hay que mapear: 1=Credit card, 2=**Cash**, 3=No charge, 4=Dispute (catálogo TLC; tu comentario "2 es cash" es correcto). |
| 2,427,612 filas × 20 columnas | Es grande: `silhouette_score` **debe** llevar `sample_size=5000`, y el scatter conviene con `s` pequeño y `alpha`. |
| ~12,125 nulos por columna (≈ 0.5 %) y 11,961 duplicados | Hay que contarlos **antes** de limpiar y mostrarlos. |
| `passenger_count` llega a 100 y `trip_distance` a 1,686 mi | Los filtros del paso 2 los quitan (pasajeros); la distancia extrema se queda (el examen prohíbe quitar atípicos por percentiles). |
| `tpep_*` son strings | No se usan; no las toques. |

**Cifras que obtuve (para que verifiques que tu notebook va bien):**
- Después de quitar duplicados + nulos: **2,196,158**; tras las 4 condiciones: **2,096,600** filas.
- `df_cash` (payment_type = Cash): **282,085** viajes.
- `passenger_count` en Cash va de 1 a 6; el 78 % son viajes de 1 pasajero.

---

## Sección 1 — Carga y exploración (7 pts)

La celda de imports/lectura ya está hecha (y `df.info()` también). **Falta**: dimensiones, columnas, `describe`, `payment_type`. Agrega estas celdas:

```python
# celda 2 (ya existe) — déjala tal cual: imports + df = pd.read_csv("../data/raw/nyc_taxi_examen.csv") + df.head(5)
```

```python
print("Dimensiones:", df.shape)
print("\nColumnas:", list(df.columns))
print("\nTipos de datos:")
print(df.dtypes)
```

```python
df.info()   # ya la tienes; puedes dejarla además de dtypes
```

```python
df.describe().round(2)          # resumen de columnas numéricas
```

```python
# Valores que existen en payment_type (incluye NaN)
print(df["payment_type"].value_counts(dropna=False))
```

> Escribe una línea en markdown/comentario: *"payment_type viene codificado en números: 1=Credit card, 2=Cash, 3=No charge, 4=Dispute; hay NaN."* Eso demuestra que "comprobaste los valores" y justifica el mapeo del paso 3.

---

## Sección 2 — Revisión y limpieza (14 pts)

Reemplaza `df_limpio = ` (esa celda está **incompleta y da SyntaxError**, hay que sustituirla):

```python
df_limpio = df.copy()

# --- ANTES de limpiar ---
print("Dimensiones antes:", df_limpio.shape)
print("\nNulos por columna:")
print(df_limpio.isnull().sum())
print("\nRegistros duplicados:", df_limpio.duplicated().sum())
```

```python
# Duplicados y nulos
df_limpio = df_limpio.drop_duplicates()
df_limpio = df_limpio.dropna()
print("Después de quitar duplicados y nulos:", df_limpio.shape)
```

```python
# Condiciones obligatorias (TODAS)
df_limpio = df_limpio[
    (df_limpio["trip_distance"] > 0) &
    (df_limpio["passenger_count"] > 0) & (df_limpio["passenger_count"] < 100) &
    (df_limpio["fare_amount"] > 0) & (df_limpio["fare_amount"] < 10000) &
    (df_limpio["trip_duration"] > 0)
].copy()

print("Dimensiones después de limpiar:", df_limpio.shape)
print("Nulos restantes:", df_limpio.isnull().sum().sum())
print("Duplicados restantes:", df_limpio.duplicated().sum())
```

Esperado: 2,427,612 → **2,096,600** filas. **No** uses percentiles ni IQR.

---

## Sección 3 — Filtrado y selección (7 pts)

Aquí está la trampa del `"Cash"`: convierte el código a texto **antes** de filtrar.

```python
# payment_type viene como número; se decodifica según el catálogo de NYC TLC
mapa_pago = {1: "Credit card", 2: "Cash", 3: "No charge", 4: "Dispute"}
df_limpio["payment_type"] = df_limpio["payment_type"].map(mapa_pago)
print(df_limpio["payment_type"].value_counts())
```

```python
df_cash = df_limpio[df_limpio["payment_type"] == "Cash"].copy()

X = df_cash[["passenger_count", "trip_distance"]].copy()

print("Viajes en efectivo:", len(df_cash))     # 282,085
X.describe().round(2)
```

Solo esas **dos** columnas entran a K-Means. `df_cash` conserva todo en unidades originales (se usa en la sección 7).

---

## Sección 4 — Transformación y estandarización (14 pts)

```python
X_log = np.log1p(X)                      # log1p a las dos variables

escalador = StandardScaler()
X_escalado = escalador.fit_transform(X_log)   # numpy array (n, 2)
```

```python
# Resumen para comprobar (medias ≈ 0, desviaciones ≈ 1)
print("Resumen de X_log (log1p):")
display(X_log.describe().round(3))

X_escalado_df = pd.DataFrame(X_escalado, columns=["passenger_count", "trip_distance"], index=X.index)
print("Resumen de X_escalado (estandarizado):")
display(X_escalado_df.describe().round(3))
```

Deja `X_escalado` como **array** (así lo usan KMeans/scatter); el DataFrame es solo para mostrar la tabla. Debes ver media 0.000 y std 1.000.

---

## Sección 5 — Evaluación de k (21 pts, la más pesada)

```python
valores_k = range(2, 7)       # 2, 3, 4, 5, 6
inercias = []
siluetas = []

for k in valores_k:
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas = modelo.fit_predict(X_escalado)
    inercias.append(modelo.inertia_)
    siluetas.append(silhouette_score(X_escalado, etiquetas,
                                     sample_size=5000, random_state=42))

resultados = pd.DataFrame({"k": list(valores_k), "Inercia": inercias, "Silueta": siluetas})
display(resultados.round(4))
```

```python
# Gráfica 1: codo (figura propia)
plt.figure(figsize=(7, 4))
plt.plot(resultados["k"], resultados["Inercia"], marker="o")
plt.title("Método del codo")
plt.xlabel("Número de clústeres (k)")
plt.ylabel("Inercia")
plt.xticks(list(valores_k))
plt.grid(alpha=0.3)
plt.show()
```

```python
# Gráfica 2: silueta (figura propia)
plt.figure(figsize=(7, 4))
plt.plot(resultados["k"], resultados["Silueta"], marker="o", color="seagreen")
plt.title("Coeficiente de silueta por k")
plt.xlabel("Número de clústeres (k)")
plt.ylabel("Silueta media")
plt.xticks(list(valores_k))
plt.grid(alpha=0.3)
plt.show()
```

Puntos fáciles de perder: dos figuras **separadas** (no subplots), título y ejes en ambas, `n_init=10`, `sample_size=5000`.

**Resultados que obtuve:**

| k | Inercia | Silueta |
|---|---|---|
| 2 | 324,582 | 0.550 |
| 3 | 160,363 | 0.612 |
| 4 | 115,076 | 0.620 |
| 5 | 86,896 | 0.625 |
| 6 | 61,145 | 0.562 |

(La silueta puede variar ±0.01 por la muestra de 5000, pero la forma es la misma.)

---

## Sección 6 — Modelo final y visualización (14 pts)

**Recomendación: k = 4.** Razones: el codo se suaviza después de 3–4; la silueta de 4 (0.620) es prácticamente igual a la máxima de k=5 (0.625), y cae en k=6; y con 4 grupos la interpretación es limpia (*solo/corto, solo/largo, acompañado/corto, acompañado/largo*). k=5 también sería defendible, pero k=4 es más parsimonioso. Lo importante es que lo justifiques con las métricas.

```python
k_final = 4

modelo_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
etiquetas = modelo_final.fit_predict(X_escalado)
centroides = modelo_final.cluster_centers_
print("Viajes por clúster:", np.bincount(etiquetas))
```

```python
plt.figure(figsize=(9, 6))
scatter = plt.scatter(X_escalado[:, 0], X_escalado[:, 1],
                      c=etiquetas, cmap="tab10", s=8, alpha=0.3,
                      vmin=-0.5, vmax=9.5)
plt.scatter(centroides[:, 0], centroides[:, 1],
            color="red", marker="X", s=250, edgecolor="black", linewidth=1.2,
            label="Centroides")
plt.title(f"K-Means (k={k_final}) — viajes en efectivo (log1p + estandarizado)")
plt.xlabel("passenger_count (log1p, estandarizado)")
plt.ylabel("trip_distance (log1p, estandarizado)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

- Eje X = `passenger_count`, eje Y = `trip_distance`, ambos transformados (no uses unidades originales aquí).
- Como `passenger_count` es discreto (1–6), los puntos se ven en **columnas verticales**; es normal. `alpha` ayuda con el traslape.
- Si tarda mucho, grafica una muestra (`np.random.RandomState(42).choice(len(X_escalado), 50000, replace=False)`), pero los centroides siempre del modelo completo.

---

## Sección 7A — Tabla de estadísticos con datos ORIGINALES (11 pts)

```python
df_resultado = df_cash.copy()
df_resultado["cluster"] = etiquetas      # mismo orden/índice que X (df_cash)

tabla = df_resultado.groupby("cluster").agg(
    viajes=("cluster", "size"),
    pasajeros_media=("passenger_count", "mean"),
    pasajeros_mediana=("passenger_count", "median"),
    distancia_media_millas=("trip_distance", "mean"),
    distancia_mediana_millas=("trip_distance", "median"),
).round(2)

display(tabla)
```

Alineación: `etiquetas` sale de `X_escalado`, que conserva el orden de `df_cash`, así que la asignación directa es correcta. **No** uses `X_log` ni `X_escalado` aquí.

**Mi resultado con k=4** (los números de cluster pueden cambiar; lo que importa son los perfiles):

| cluster | viajes | pas. media | pas. mediana | dist. media | dist. mediana |
|---|---|---|---|---|---|
| A | 160,372 | 1.00 | 1 | 1.36 | 1.20 |
| B | 59,946 | 1.00 | 1 | 9.55 | 7.96 |
| C | 18,849 | 2.62 | 2 | 12.08 | 10.37 |
| D | 42,918 | 2.55 | 2 | 1.53 | 1.30 |

---

## Sección 7B — Preguntas (12 pts, 3 c/u)

Escribe en tus palabras, con números **de tu tabla** (ajusta los números de cluster a los que te salgan). Borradores:

**P1 — Selección de k.**
Elegí k=4. La inercia siempre baja al aumentar k (de 324,582 en k=2 a 61,145 en k=6), así que por sí sola no decide; el codo se aplana a partir de k=3–4. La silueta sube de 0.550 (k=2) a 0.612 (k=3) y 0.620 (k=4), alcanza su máximo en k=5 (0.625) con una mejora mínima y cae a 0.562 en k=6, donde los grupos se fragmentan. Con k=4 la silueta es casi la máxima y los grupos son claros e interpretables (solitarios/acompañados × cortos/largos), mientras que k=5 solo añade un subgrupo pequeño sin ganancia relevante.

**P2 — Interpretación de grupos.**
- *Solitarios de corta distancia* (≈160 mil viajes, el más grande): mediana 1 pasajero, mediana ≈1.2 mi; el viaje típico urbano.
- *Solitarios de larga distancia* (≈60 mil): mediana 1 pasajero, mediana ≈8 mi.
- *Grupos pequeños de corta distancia* (≈43 mil): media ≈2.55 pasajeros, mediana 2, mediana ≈1.3 mi.
- *Grupos de larga distancia* (≈19 mil, el más pequeño): media ≈2.6 pasajeros, mediana 2, mediana ≈10.4 mi, media ≈12 mi.
Comenta también que la media de distancia es mayor que la mediana (cola derecha por viajes largos).

**P3 — Menor vs mayor promedio de pasajeros.**
Los grupos con ≈1 pasajero concentran la mayoría de los viajes (≈220 mil de 282 mil) frente a los de ≈2.6 pasajeros (≈62 mil). Entre los de más pasajeros, la distancia depende del subgrupo (1.3 mi vs 10.4 mi de mediana): tener más pasajeros no define por sí solo la distancia. Compara filas de tu tabla: mismo nivel de pasajeros, distancias muy distintas.

**P4 — ¿Más pasajeros = mayor distancia?**
No siempre. El grupo de ≈2.55 pasajeros recorre solo ≈1.3 mi de mediana, parecido a los solitarios cortos (1.2 mi), y los solitarios largos (≈8 mi) recorren más que los grupos cortos con más pasajeros. Solo el grupo de viajes largos con ≈2.6 pasajeros combina ambas cosas. Por tanto la relación no es general: existen viajes cortos y largos tanto para 1 pasajero como para varios, y no todos se comportan igual.

---

## Checklist de entrega

- [ ] Reemplazar `df_limpio = ` (error de sintaxis) y que **todas** las celdas corran en orden (Kernel → Restart & Run All).
- [ ] Evidencia visible: `shape`, columnas, dtypes, describe, value_counts, nulos/duplicados antes y shape antes/después.
- [ ] `payment_type` mapeado a texto **antes** de filtrar `"Cash"`.
- [ ] `X` con solo 2 columnas; `X_log` (`np.log1p`); `X_escalado` (`StandardScaler`).
- [ ] Tabla k/Inercia/Silueta + **dos** gráficas separadas con títulos y ejes.
- [ ] Scatter con centroides **X** contrastantes y leyenda.
- [ ] Tabla de estadísticos sobre `df_cash` original, redondeada a 2 decimales.
- [ ] 4 respuestas escritas debajo de cada "**Respuesta:**" (celda markdown, con números de tu tabla).
- [ ] Guardar con el nombre `Examen_B_Apellido_Nombre.ipynb` (tu archivo actual es `Examen_B_Martínez_Aarón.ipynb`; confirma que es el formato aceptado).
- [ ] Cuidar el tiempo (90 min): lectura + limpieza del CSV de 2.4M filas tarda unos segundos; el ciclo de k tarda ~1–2 min.
