import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.getcwd())

ARCHIVO = r"Base_de_datos_Robusta.csv"

CARRERA_OBJETIVO  = "PROGRAMACIÓN"
SEMESTRE_OBJETIVO = "6"           

MATERIA_OBJETIVO  = ""             

cal_cols  = ["PERIODO 1","PERIODO 2","PERIODO 3","CALIFICACION"]
asis_cols = ["ASISTENCIA PERIODO 1","ASISTENCIA PERIODO 2","ASISTENCIA PERIODO 3","TOTAL DE ASITENCIAS"]
min_filas = 5

try:
    df = pd.read_csv(ARCHIVO, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(ARCHIVO, encoding="utf-8-sig")

df["CARRERA"] = df["CARRERA"].astype(str).str.strip().str.upper()
df["NOMBRE ASIGNATURA"] = df["NOMBRE ASIGNATURA"].astype(str).str.strip().str.upper()
df["SEMESTRE"] = df["SEMESTRE"].astype(str).str.strip()

m = df["DESERTO"].astype(str).str.strip().str.upper()
df["DESERTO"] = np.where(m.isin(["1","SI","TRUE","VERDADERO"]), 1,
                  np.where(m.isin(["0","NO","FALSE","FALSO"]), 0,
                           pd.to_numeric(df["DESERTO"], errors="coerce")))

for c in cal_cols + asis_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

base = df[(df["CARRERA"] == CARRERA_OBJETIVO.upper()) &
          (df["SEMESTRE"] == str(SEMESTRE_OBJETIVO))].copy()

if base.empty:
    print("No hay registros para esa CARRERA/SEMESTRE. Verifica valores.")
    print("\nEjemplos disponibles:")
    print("Carreras:", sorted(df["CARRERA"].dropna().unique())[:10], "...")
    print("Semestres:", sorted(df["SEMESTRE"].dropna().unique()))
    raise SystemExit

if MATERIA_OBJETIVO.strip() == "":
    materias = sorted(base["NOMBRE ASIGNATURA"].dropna().unique())
else:
    materias = [MATERIA_OBJETIVO.strip().upper()]

print("\nMaterias que se graficarán:")
for i, mat in enumerate(materias, 1):
    print(f"{i:>2}. {mat}")

def grafica_materia(df_mat, carrera, semestre, materia):
    # 1) usar TODAS las columnas numéricas posibles + DESERTO (sin filtrar por variación)
    cols_all = [c for c in cal_cols + asis_cols if c in df_mat.columns] + ["DESERTO"]

    # 2) DESERTO debe tener 0 y 1 para que tenga sentido correlacionar
    if "DESERTO" not in df_mat.columns or df_mat["DESERTO"].dropna().nunique() <= 1:
        print(f"{carrera} | Sem {semestre} | {materia}: DESERTO es constante (no hay 0 y 1). Se omite.")
        return

    # 3) mínimo de filas (cuenta con todas las columnas candidatas)
    if df_mat.dropna(how="all", subset=cols_all).shape[0] < min_filas:
        print(f"⚠️  {carrera} | Sem {semestre} | {materia}: datos insuficientes para correlación.")
        return

    # 4) correlación Spearman y RELLENAR NaN con 0
    corr = df_mat[cols_all].corr(method="spearman").fillna(0)

    # 5) salida consola: correlación vs DESERTO
    print(f"\n {carrera} | Semestre {semestre} | {materia}")
    print(corr["DESERTO"].sort_values(ascending=False).to_string())

    # 6) heatmap (ahora las columnas constantes aparecen con 0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidth=.5, square=True)
    plt.title(f"Semestre {semestre} | {carrera} | {materia}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


for materia in materias:
    df_materia = base[base["NOMBRE ASIGNATURA"] == materia].copy()
    if df_materia.empty:
        print(f"⚠️  Materia no encontrada en el filtro actual: {materia}")
        continue
    grafica_materia(df_materia, CARRERA_OBJETIVO.upper(), SEMESTRE_OBJETIVO, materia)
