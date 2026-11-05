# Optimal_Route: Algoritmos de Planificación de Rutas y Tratamiento de Datos Planetarios

[![NASA Space Apps Challenge](https://img.shields.io/badge/NASA%20Space%20Apps-2024-0B3D91?style=for-the-badge&logo=nasa&logoColor=white)](https://www.spaceappschallenge.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit Quantum](https://img.shields.io/badge/Qiskit-QAOA%20Optimization-6929C4?style=for-the-badge&logo=qiskit&logoColor=white)](https://qiskit.org/)
[![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-Interactive%20Science-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

Repositorio científico y pedagógico oficial de los algoritmos de navegación, tratamiento de datos satelitales y optimización cuántica.

Este repositorio contiene la fundamentación matemática, el código fuente en Python y una suite de **6 libretas interactivas (`.ipynb`)** totalmente reproducibles que van desde la decodificación de datos altimétricos brutos hasta la resolución variacional con computación cuántica.

---

## Índice Pedagógico de Libretas

| Libreta | Título | Descripción Técnica |
| :--- | :--- | :--- |
| [**`01`**](notebooks/01_pipeline_y_tratamiento_dem.ipynb) | **Pipeline y Tratamiento de Datos DEM** | Decodificación de altimetría Float32 (MOLA/CTX), georreferenciación en `IAU_2015:49900`, consultas continuas $O(1)$ por interpolación bilineal y operador de gradiente por diferencias finitas. |
| [**`02`**](notebooks/02_biofisica_clima_y_capas_multimision.ipynb) | **Biofísica, Clima MEDA y Capas Multi-Misión** | Modelo metabólico de Pandolf/Margaria en gravedad marciana ($3.71 \text{ m/s}^2$), balance de $O_2$ (PLSS), capas CRISM (filosilicatos/carbonatos), THEMIS (inercia térmica) y efemérides solares con atenuación de Beer-Lambert ($\tau$). |
| [**`03`**](notebooks/03_planificacion_clasica_astar.ipynb) | **Planificación Clásica con $A^*$** | Grafo 2.5D ponderado fisiológicamente, min-heap binario ($O(\|E\| \log \|V\|)$), demostración de heurística admisible y consistente en $O_2$, y generación de perfiles longitudinales 2D. |
| [**`04`**](notebooks/04_optimizacion_cuantica_qubo_qaoa.ipynb) | **Optimización Cuántica (QUBO / QAOA)** | Corredor variacional longitudinal-transversal, formulación del Hamiltoniano de Costo $\hat{H}_C$ e Ising, circuito cuántico QAOA parametrizado con Qiskit y minimización de varianza de pendiente (isoclinas). |
| [**`05`**](notebooks/05_benchmark_comparativo_astar_vs_qaoa.ipynb) | **Benchmark Comparativo: $A^*$ vs. QAOA** | Matriz de evaluación cuantitativa: distancia ($m$), consumo de $O_2$ ($L$), pendiente máxima ($^\circ$), rugosidad de la marcha, y escalabilidad algorítmica clásica vs. procesadores cuánticos NISQ. |
| [**`06`**](notebooks/06_misiones_avanzadas_tsp_y_evacuacion.ipynb) | **Misiones Avanzadas: TSP y Evacuación** | Detección geomorfométrica automática de POIs desde el DEM (criterios NASA Landing Site), Agente Viajero (TSP) con presupuesto de $O_2$ y circuito de retorno, y evacuación reactiva de emergencia ante tormentas de polvo MEDA. |

---

## Fundamentación Matemática y Física

### 1. Modelo de Consumo Metabólico ($PLSS$)
Adaptación para gravedad marciana ($g = 3.71\text{ m/s}^2$) y trajes espaciales presurizados:
$$\dot{V}_{O_2} = \dot{V}_{\text{base}} \times f_{\text{pendiente}}(\theta) \times C_{\text{climb}}\left(\frac{\Delta z}{d}\right) \times M_{\text{run}} \times P_{\text{termico}}$$
- **Tasa basal:** $\dot{V}_{\text{base}} = 50\text{ L/h}$ ($\approx 0.0139\text{ L/s}$) a marcha nominal ($1.2\text{ m/s}$).
- **Umbral de seguridad biomecánica:** $\theta_{\text{safe}} = 25^\circ$. Pendientes superiores disparan penalización cuadrática estricta para evitar resbalamientos y rotura del traje.

### 2. Operador de Gradiente Discreto (Pendiente Real no Exagerada)
$$\nabla z = \left( \frac{\partial z}{\partial x}, \frac{\partial z}{\partial z} \right) \approx \left( \frac{z(x + \Delta x, z) - z(x - \Delta x, z)}{2 \Delta x}, \frac{z(x, z + \Delta z) - z(x, z - \Delta z)}{2 \Delta z} \right)$$
$$\theta^\circ = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial z}\right)^2}\right) \times \frac{180^\circ}{\pi}$$

### 3. Heurística Admisible para $A^*$
$$h(n) = \left( \frac{\|\mathbf{p}_{\text{meta}} - \mathbf{p}_n\|_2}{v_{\text{walk}}} \right) \cdot \dot{V}_{\text{base}} \cdot 0.85 \leq h^*(n)$$
Garantiza matemáticamente encontrar el camino de menor gasto metabólico en la rejilla.

### 4. Formulación Cuántica QUBO / QAOA
Definido sobre un corredor de $S$ rebanadas longitudinales y $K$ estados discretos transversales ($S \times K$ qubits):
$$\hat{H}_C = \sum_{s=1}^{S} \sum_{i, j} J_{s, i, j} \hat{Z}_{s-1, i} \hat{Z}_{s, j} + \sum_{s, i} h_{s, i} \hat{Z}_{s, i}$$
$$\hat{H}_B = \sum_{s, k} \hat{X}_{s, k}$$
Estado variacional cuántico de $p$ capas:
$$|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{l=1}^p e^{-i \beta_l \hat{H}_B} e^{-i \gamma_l \hat{H}_C} |+\rangle^{\otimes N}$$
*Ventaja:* QAOA minimiza la varianza de la pendiente contorneando curvas de nivel con menor fatiga física.

---

## Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/AngelGab3/Optimal_Route.git
cd Optimal_Route
```

### 2. Crear entorno virtual e instalar dependencias
```bash
# Con Conda
conda create -n mars_route python=3.11 -y
conda activate mars_route
pip install -r requirements.txt

# O con venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Lanzar Jupyter Lab / Notebooks
```bash
jupyter lab
```
Abre la carpeta `notebooks/` y ejecuta las libretas en orden secuencial (`01` a `06`).

---

## Estructura del Repositorio

```
Optimal_Route/
├── README.md               # Este documento
├── requirements.txt        # Dependencias de Python
├── data/
│   ├── jezero_test/        # DEM real del Cráter Jezero (MOLA/CTX 20 m/px)
│   └── synth_crater/       # DEM sintético para validación rápida
├── src/                    # Módulos Python
│   ├── dem_pipeline.py     # Lector Float32 y muestreo O(1)
│   ├── physiology.py       # Modelo bioenergético y O2
│   ├── multimission.py     # Derivación CRISM y THEMIS
│   ├── solar_physics.py    # Efemérides Keplerianas y atenuación MEDA
│   ├── astar.py            # Planificador clásico A*
│   ├── qaoa_router.py      # Planificador cuántico QUBO/QAOA
│   └── mission_planner.py  # Algoritmo TSP y evacuación
└── notebooks/              # Suite de 6 libretas pedagógicas (.ipynb)
```

---

## Misiones NASA Referenciadas
- **Mars 2020:** Rover *Perseverance* & Estación Meteorológica *MEDA*.
- **MRO (Mars Reconnaissance Orbiter):** Espectrómetro *CRISM* y Cámaras *CTX / HiRISE*.
- **Mars Odyssey:** Cámara de Emisión Térmica *THEMIS*.
- **MGS (Mars Global Surveyor):** Altimetría Láser *MOLA*.
