import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ARCHIVO = r"Base_de_datos_Robusta.csv"

SEMESTRE_OBJETIVO = "6"      
MATERIA_OBJETIVO = ""       

MIN_FILAS = 5

cal_cols  = ["PERIODO 1","PERIODO 2","PERIODO 3","CALIFICACION"]
asis_cols = ["ASISTENCIA PERIODO 1","ASISTENCIA PERIODO 2","ASISTENCIA PERIODO 3","TOTAL DE ASITENCIAS"]
acad_cols = cal_cols + asis_cols

try:
    df = pd.read_csv(ARCHIVO, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(ARCHIVO, encoding="utf-8-sig")

df["NOMBRE ASIGNATURA"] = df["NOMBRE ASIGNATURA"].astype(str).str.strip().str.upper()
df["SEMESTRE"] = df["SEMESTRE"].astype(str).str.strip()

s = pd.to_numeric(df["SEMESTRE DE INTERRUPCION"], errors="coerce")
df["SEM_BIN"] = np.where(s == 6, 0, np.where(s.isin([1,2,3,4,5]), 1, 0))

cols_all = [c for c in acad_cols if c in df.columns] + ["SEM_BIN"]
for c in cols_all:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

if SEMESTRE_OBJETIVO.strip():
    semestres = [SEMESTRE_OBJETIVO.strip()]
else:
    sem_vals = pd.to_numeric(df["SEMESTRE"], errors="coerce")
    order = sem_vals.sort_values().dropna().astype(int).astype(str).unique().tolist()
    non_num = df.loc[sem_vals.isna(), "SEMESTRE"].dropna().unique().tolist()
    semestres = order + [s for s in non_num if s not in order]

print("\nSemestres que se procesarán:")
for i, sem in enumerate(semestres, 1):
    print(f"{i:>2}. {sem}")

def grafica_materia(df_mat, materia, semestre):
    data = df_mat[cols_all].copy()
    if len(data) < MIN_FILAS:
        print(f"{materia} (Sem {semestre}): pocos registros ({len(data)}).")
        return

    corr = data.corr(method="spearman").fillna(0)
    for i in range(len(corr)):
        corr.iat[i, i] = 1.0

    print(f"\n{materia} — Semestre {semestre} — correlación con SEM_BIN:")
    print(corr["SEM_BIN"].sort_values(ascending=False).to_string())

    n = len(corr.columns)
    plt.figure(figsize=(max(12, 0.7*n), max(10, 0.6*n)))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, square=True, cbar=True)
    plt.title(f"Matriz Spearman — {materia} | Sem {semestre}", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout(pad=3.0)
    plt.show()

for semestre in semestres:
    base = df[df["SEMESTRE"] == semestre].copy()
    if base.empty:
        print(f"No hay registros para Semestre {semestre}.")
        continue

    if MATERIA_OBJETIVO.strip():
        materias = [MATERIA_OBJETIVO.strip().upper()]
    else:
        materias = sorted(base["NOMBRE ASIGNATURA"].dropna().unique())

    print(f"\nMaterias a graficar en Semestre {semestre}:")
    for i, mat in enumerate(materias, 1):
        print(f"{i:>2}. {mat}")

    for materia in materias:
        sub = base[base["NOMBRE ASIGNATURA"] == materia].copy()
        if sub.empty:
            print(f"Materia no encontrada en Sem {semestre}: {materia}")
            continue
        grafica_materia(sub, materia, semestre)
