# Examen práctico – Primer parcial de Tópicos de Big Data
## Guía de actividades y preguntas de análisis por dataset (Our World in Data)

Este documento contiene la estructura del examen y las 10 preguntas de análisis para
cada uno de los 5 datasets. Cada alumno trabajará con **un solo dataset**, asignado
aleatoriamente, pero **todos responden el mismo tipo de preguntas**, adaptadas a su
dataset específico. Esto asegura equidad en la dificultad y reduce la posibilidad de copia.

---

## 0. Estructura común del notebook (para todos los datasets)


Independientemente del dataset asignado, el notebook de cada alumno debe seguir esta
misma estructura de pasos, antes de llegar a las 10 preguntas de análisis:

1. **Carga de datos**: usar el código de fetch (pandas + requests) correspondiente a su dataset.
2. **Exploración inicial**: `.head()`, `.info()`, `.describe()`, tipos de datos, `.shape`, conteo de valores nulos.
3. **Limpieza/filtrado inicial** (ver nota importante abajo): eliminar filas que no son países individuales (regiones, continentes, agrupaciones de ingreso, "World", etc.)
4. **Las 9 preguntas de análisis** (detalladas abajo por dataset).
5. **Conclusión breve** (5-10 líneas): qué aprendieron del dataset asignado.

### ⚠️ Nota importante común a varios datasets: filas "agregadas" son filas de regiones o zonas

Los CSV de OWID no solo contienen países individuales: también incluyen filas para
continentes, agrupaciones por nivel de ingreso, regiones de la ONU, y "World". Si no se
filtran, contaminan cualquier comparación entre países o cálculo estadístico. Ya
verifiqué cuál es la forma correcta de identificarlas y filtrarlas en cada dataset (ver
sección específica de cada uno). En general, la regla es:

- Si el dataset tiene columna `owid_region`: los países reales tienen un valor no nulo
  ahí; los agregados (continentes, etc.) tienen ese campo vacío.
- Si el dataset **no** tiene `owid_region`: los países reales tienen un código ISO de 3
  letras en la columna `code`; los agregados tienen `code` vacío (o `NaN`).

---

## 1. Dataset: Consumo de alcohol (adultos que bebieron en el último año)

**Código de carga:**
```python
import pandas as pd
import requests

df = pd.read_csv(
    "https://ourworldindata.org/grapher/share-of-adults-who-drank-alcohol-in-last-year.csv?v=1&csvType=full&useColumnShortNames=true",
    storage_options={'User-Agent': 'Our World In Data data fetch/1.0'}
)
metadata = requests.get(
    "https://ourworldindata.org/grapher/share-of-adults-who-drank-alcohol-in-last-year.metadata.json?v=1&csvType=full&useColumnShortNames=true"
).json()
```

**Columnas:** `entity`, `code`, `year`, `alcohol__consumers_past_12_months__pct__age_standardized__sex_both_sexes`
**Unidad:** % de adultos (15+) que bebieron alcohol en los últimos 12 meses (estandarizado por edad)
**Rango de años:** 2000–2020
**Filtrado de agregados:** usar `df['code'].notna()`

### Preguntas de análisis

**Exploración y filtrado (2)**
1. Filtra el dataset para quedarte solo con países reales (código ISO no nulo). ¿Cuántos países distintos hay? Para el año más reciente disponible (2020), muestra los 10 países con mayor porcentaje de consumo y los 10 con menor porcentaje.
2. Elige 4 países de tu interés y filtra el DataFrame para mostrar únicamente su información entre 2010 y 2020. Usa indexación booleana combinando `.isin()` y comparación de años.

**Comparación entre países/grupos (2)**
3. Compara el porcentaje de consumo de alcohol entre 5 países que tú elijas (de distintos continentes) en el año 2019. ¿Cuál tiene el valor más alto y cuál el más bajo? ¿Qué diferencia porcentual hay entre ellos?
4. Define dos grupos de países con una lista propia (por ejemplo, "Europa" vs "Latinoamérica", usando 5 países representativos de cada uno). Calcula el promedio de consumo de alcohol de cada grupo en el año más reciente disponible y compáralos.

**Tendencia temporal (2)**
5. Para un país de tu elección, grafica (o calcula) cómo cambió el porcentaje de consumo de alcohol entre 2000 y 2020. ¿Aumentó o disminuyó? ¿En qué año tuvo su valor máximo y en cuál el mínimo?
6. Calcula el cambio porcentual entre el primer y el último año disponible para los 5 países del punto 3. ¿Cuál tuvo el mayor incremento y cuál la mayor caída?

**Cálculos estadísticos con NumPy (1)**
7. Usando NumPy, calcula la media, mediana y desviación estándar del porcentaje de consumo de alcohol a nivel mundial (todos los países) para el año más reciente. Interpreta brevemente qué te dice la desviación estándar sobre la variabilidad entre países.

**Visualización (1)**
8. Crea una gráfica de líneas con matplotlib que muestre la evolución del consumo de alcohol (2000–2020) para 3 países de tu elección en una sola figura, con leyenda y etiquetas de ejes. Escribe 2-3 líneas interpretando lo que se observa.

**Integradora (1)**
9. Crea dos gráficas de barras horizontales con el Top 10 de países con mayor consumo de alcohol, la primera en el año de tu nacimiento, la segunda en el año 2023. Interpreta el resultado: ¿qué patrón observas?

---

## 2. Dataset: Población total de inmigrantes internacionales (1990–2024)

**Código de carga:**
```python
import pandas as pd
import requests

df = pd.read_csv(
    "https://ourworldindata.org/grapher/migrant-populations.csv?v=1&csvType=full&useColumnShortNames=true&metric=immigrants&unit=number",
    storage_options={'User-Agent': 'Our World In Data data fetch/1.0'}
)
metadata = requests.get(
    "https://ourworldindata.org/grapher/migrant-populations.metadata.json?v=1&csvType=full&useColumnShortNames=true&metric=immigrants&unit=number"
).json()
```

**Columnas:** `entity`, `code`, `year`, `immigrants_all`, `owid_region`
**Unidad:** número absoluto de inmigrantes internacionales residiendo en el país
**Rango de años:** 1990–2024, en pasos de 5 años (1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024)
**Filtrado de agregados:** usar `df['owid_region'].notna()` (los continentes y agrupaciones como "World", "Europe", "High-income countries", etc. tienen `owid_region` vacío)

### Preguntas de análisis

**Exploración y filtrado (2)**
1. Filtra el dataset para quedarte solo con países reales (usa la columna `owid_region`). Para el año 2024, muestra los 10 países con mayor número absoluto de inmigrantes.
2. Elige un continente (usando la columna `owid_region`, ej. `"Europe"`, `"Asia"`, `"Africa"`) y filtra todos los países de ese continente en el año 2024. ¿Cuántos países tiene ese grupo en el dataset?

**Comparación entre países/grupos (2)**
3. Compara el número de inmigrantes en 2024 entre 5 países que tú elijas. ¿Cuál recibe más migrantes en términos absolutos? Ten en cuenta que esto no está normalizado por población total, ¿qué limitación tiene esta comparación?
4. Usando la columna `owid_region`, calcula el promedio de inmigrantes por país en cada continente para el año 2024 con `.groupby()`. ¿Qué continente tiene, en promedio, más inmigrantes por país?

**Tendencia temporal (2)**
5. Para un país de tu elección, muestra cómo cambió su número de inmigrantes entre 1990 y 2024. ¿Hay algún salto brusco entre dos periodos consecutivos? Investiga brevemente si coincide con algún evento migratorio conocido.
6. Calcula el cambio porcentual de inmigrantes entre 1990 y 2024 para 5 países de tu elección. ¿Cuál tuvo el mayor crecimiento relativo?

**Cálculos estadísticos con NumPy (1)**
7. Usando NumPy, calcula la media, mediana y desviación estándar del número de inmigrantes entre todos los países (reales, sin agregados) para el año 2024. Dado que esta variable suele tener una distribución muy asimétrica (pocos países con valores enormes), compara la media contra la mediana e interpreta la diferencia.

**Visualización (1)**
8. Crea una gráfica de líneas que muestre la evolución del número de inmigrantes (1990–2024) para 3 países de tu elección, en una sola figura con leyenda. Interpreta brevemente qué observas.

**Integradora (1)**
9. Crea dos gráficas de barras del top 10 de paises con el total de inmigrantes por pais, la primera en el año de tu nacimiento, la segunda en el año 2023. Interpreta el resultado: ¿qué paí concentra más migración recibida?

---

## 3. Dataset: Número de personas subalimentadas (desnutrición)

**Código de carga:**
```python
import pandas as pd
import requests

df = pd.read_csv(
    "https://ourworldindata.org/grapher/number-undernourished.csv?v=1&csvType=full&useColumnShortNames=true",
    storage_options={'User-Agent': 'Our World In Data data fetch/1.0'}
)
metadata = requests.get(
    "https://ourworldindata.org/grapher/number-undernourished.metadata.json?v=1&csvType=full&useColumnShortNames=true"
).json()
```

**Columnas:** `entity`, `code`, `year`, `_2_1_1_number_of_undernourished_people__000000000024001__value__006132__millions`
(sugerencia: renombrar esta columna a algo como `undernourished` con `df.rename()` al inicio, ya que el nombre original es poco práctico)
**Unidad:** número absoluto de personas subalimentadas (aunque el nombre de columna dice "millions", los valores ya vienen en unidades absolutas de personas)
**Rango de años:** 2000/2001–2024 (⚠️ el dataset tiene **muchos años faltantes por país**: varios países solo tienen datos hasta cierto año o con huecos, esto es intencional para que trabajen con `NaN`)
**Filtrado de agregados:** usar `df['code'].notna()` (agregados como "Africa (FAO)", "Asia (FAO)", "World", etc. tienen `code` vacío)

### Preguntas de análisis

**Exploración y filtrado (2)**
1. Filtra el dataset para quedarte solo con países reales (código ISO no nulo). ¿Qué porcentaje de las filas originales correspondía a agregados regionales? Para el año más reciente disponible por país, identifica los 10 países con mayor número de personas subalimentadas.
2. Este dataset tiene datos faltantes irregulares. Elige un país y muestra qué años tienen dato disponible y cuáles no, usando `.isna()`. ¿Cuál es el rango de años con información continua para ese país?

**Comparación entre países/grupos (2)**
3. Compara el número de personas subalimentadas entre 5 países que elijas, usando el año más reciente disponible para cada uno (no necesariamente el mismo año para todos, dado los huecos). Documenta qué año usaste para cada país.
4. Define dos grupos de países (por ejemplo, un grupo de África Subsahariana y otro de Asia, usando listas de 5 países cada uno) y compara el promedio de personas subalimentadas entre ambos grupos en 2022.

**Tendencia temporal (2)**
5. Para un país con datos relativamente completos (ej. India, Nigeria, Brasil, Etiopía), muestra la evolución de personas subalimentadas a lo largo del tiempo disponible. ¿La tendencia es creciente, decreciente o fluctuante?
6. Calcula el cambio absoluto y porcentual entre el primer y el último año con dato disponible, para 5 países de tu elección.

**Cálculos estadísticos con NumPy (1)**
7. Usando NumPy, calcula media, mediana y desviación estándar del número de personas subalimentadas entre todos los países (sin `NaN`) en el año 2022. Ten cuidado de excluir los valores faltantes antes de calcular (`np.nanmean`, `np.nanstd`, o filtrando con `.dropna()`).

**Visualización (1)**
8. Grafica con matplotlib la evolución temporal de 3 países de tu elección en una misma figura. Ya que hay huecos en los datos, asegúrate de que la gráfica no una puntos con años faltantes de forma engañosa (puedes usar `marker='o'` para visualizar los puntos reales).

**Integradora (1)**
9. Crea dos gráficas de barras con el Top 10 de países con mayor número de personas subalimentadas, en la primera en el año de tu nacimiento, la segunda en el año 2023. Interpreta: ¿coincide este ranking con lo que esperarías según población total de cada país? 

---

## 4. Dataset: PIB per cápita (Gross Domestic Product per capita)

**Código de carga:**
```python
import pandas as pd
import requests

df = pd.read_csv(
    "https://ourworldindata.org/grapher/gdp-per-capita-worldbank-constant-usd.csv?v=1&csvType=full&useColumnShortNames=true",
    storage_options={'User-Agent': 'Our World In Data data fetch/1.0'}
)
metadata = requests.get(
    "https://ourworldindata.org/grapher/gdp-per-capita-worldbank-constant-usd.metadata.json?v=1&csvType=full&useColumnShortNames=true"
).json()
```

**Columnas:** `entity`, `code`, `year`, `ny_gdp_pcap_kd`
**Unidad:** dólares constantes de 2015 (ajustado por inflación, no por poder adquisitivo)
**Rango de años:** 1960–2025 (el rango más amplio de los 5 datasets)
**Filtrado de agregados:** usar `df['code'].notna()`

### Preguntas de análisis

**Exploración y filtrado (2)**
1. Filtra el dataset para quedarte solo con países reales. Para el año 2023 (o el último año disponible), muestra los 10 países con mayor PIB per cápita y los 10 con menor PIB per cápita.
2. Filtra el DataFrame para un país de tu elección y muestra únicamente los años en que su PIB per cápita fue mayor a un umbral que tú definas (por ejemplo, mayor a $20,000). Usa indexación booleana.

**Comparación entre países/grupos (2)**
3. Compara el PIB per cápita en el año más reciente entre 5 países de tu elección de distintos continentes. Calcula cuántas veces más rico es el país con mayor PIB per cápita respecto al de menor.
4. Define dos grupos de países con listas propias (por ejemplo, "economías desarrolladas" vs "economías emergentes", 5 países cada uno) y compara el promedio de PIB per cápita de cada grupo en 2023.

**Tendencia temporal (2)**
5. Para un país de tu elección, grafica o calcula la evolución de su PIB per cápita desde 1990 hasta el año más reciente. ¿Hay caídas notorias? Investiga brevemente si coinciden con crisis económicas conocidas (ej. 2008-2009).
6. Calcula la tasa de crecimiento porcentual del PIB per cápita entre 2000 y el año más reciente disponible, para 5 países de tu elección. ¿Cuál creció más en términos relativos?

**Cálculos estadísticos con NumPy (1)**
7. Usando NumPy, calcula la media, mediana y desviación estándar del PIB per cápita mundial (todos los países) en el año más reciente. Interpreta la diferencia entre media y mediana: ¿qué indica sobre la desigualdad económica global?

**Visualización (1)**
8. Crea una gráfica de líneas con matplotlib mostrando la evolución del PIB per cápita (1990–actualidad) de 3 países de tu elección en una misma figura, con etiquetas y leyenda. Interpreta brevemente las diferencias en las trayectorias.

**Integradora (1)**
9. Crea dos gráficas de barras horizontales con el Top 10 de países con mayor PIB per cápita, la primera en el año de tu nacimiento, la segunda en el año 2023. Interpreta: ¿el ranking coincide con las economías que consideras "más grandes" del mundo (por PIB total) que conoces? 

---

## 5. Dataset: Homicidios (UNODC)

**Código de carga:**
```python
import pandas as pd
import requests

df = pd.read_csv(
    "https://ourworldindata.org/grapher/homicides-unodc.csv?v=1&csvType=full&useColumnShortNames=true",
    storage_options={'User-Agent': 'Our World In Data data fetch/1.0'}
)
metadata = requests.get(
    "https://ourworldindata.org/grapher/homicides-unodc.metadata.json?v=1&csvType=full&useColumnShortNames=true"
).json()
```

**Columnas:** `entity`, `code`, `year`, `value__category_total__sex_total__age_total__unit_of_measurement_counts`, `owid_region`
(sugerencia: renombrar la columna de valores a `homicides` para trabajar más cómodo)
**Unidad:** número absoluto de homicidios intencionales al año
**Rango de años:** 1990–2024
**Filtrado de agregados:** usar `df['owid_region'].notna()`

### Preguntas de análisis

**Exploración y filtrado (2)**
1. Filtra el dataset para quedarte solo con países reales (usa `owid_region`). Para el año 2024 (o el último disponible), muestra los 10 países con mayor número absoluto de homicidios.
2. Filtra el DataFrame para mostrar únicamente los registros de un país de tu elección entre 2010 y 2024, usando indexación booleana combinando condición de país y rango de años.

**Comparación entre países/grupos (2)**
3. Compara el número de homicidios en el año más reciente entre 5 países de tu elección. Comenta qué limitación tiene comparar el número absoluto sin considerar el tamaño de la población de cada país.
4. Usando la columna `owid_region`, calcula con `.groupby()` el total de homicidios por continente en el año más reciente disponible. ¿Qué continente concentra más homicidios en términos absolutos?

**Tendencia temporal (2)**
5. Para un país de tu elección, muestra la evolución del número de homicidios entre 1990 y 2024. ¿La tendencia es creciente, decreciente o fluctuante? ¿Hay algún pico notable?
6. Calcula el cambio porcentual en el número de homicidios entre 2000 y el año más reciente, para 5 países de tu elección. ¿Cuál tuvo la mayor reducción y cuál el mayor incremento?

**Cálculos estadísticos con NumPy (2)**
7. Usando NumPy, calcula media, mediana y desviación estándar del número de homicidios entre todos los países en el año más reciente. Interpreta la diferencia entre media y mediana.

**Visualización (1)**
8. Crea una gráficas de líneas con matplotlib mostrando la evolución del número de homicidios (1990–2024) de 3 países de tu elección en una misma figura, con leyenda y etiquetas de ejes.

**Integradora (1)**
9. Crea dos gráficas de barras con el top 10 por pais del total de homicidios, la primera en el año de tu nacimiento, la segunda en el año 2023. Interpreta el resultado y comenta por qué esta variable en términos absolutos puede dar una imagen distorsionada de qué tan "violento" es realmente un país o región.

---
