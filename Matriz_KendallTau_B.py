import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

ARCHIVO = r"Base_de_datos_Robusta.csv"

SEMESTRE_OBJETIVO = "5"      
MATERIA_OBJETIVO = ""       

MIN_FILAS = 5

cal_cols  = ["PERIODO 1","PERIODO 2","PERIODO 3","CALIFICACION"]
asis_cols = ["ASISTENCIA PERIODO 1","ASISTENCIA PERIODO 2","ASISTENCIA PERIODO 3","TOTAL DE ASISTENCIAS"]
acad_cols = cal_cols + asis_cols

COL_SEM_INT = "SEMESTRE DE INTERRUPCION"

try:
    df = pd.read_csv(ARCHIVO, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(ARCHIVO, encoding="utf-8-sig")

df["NOMBRE ASIGNATURA"] = df["NOMBRE ASIGNATURA"].astype(str).str.strip().str.upper()
df["SEMESTRE"] = df["SEMESTRE"].astype(str).str.strip()

cols_all = [c for c in acad_cols if c in df.columns] + [COL_SEM_INT]
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

def kendall_matrix(data: pd.DataFrame) -> pd.DataFrame:
    cols = list(data.columns)
    n = len(cols)
    K = np.zeros((n, n), dtype=float)

    for i in range(n):
        xi = data[cols[i]].values
        for j in range(i, n):
            if i == j:
                K[i, j] = 1.0
            else:
                yj = data[cols[j]].values
                tau, p = kendalltau(xi, yj)
                if np.isnan(tau):
                    tau = 0.0
                K[i, j] = tau
                K[j, i] = tau
    out = pd.DataFrame(K, index=cols, columns=cols)
    return out

def grafica_materia(df_mat: pd.DataFrame, materia: str, semestre: str):
    data = df_mat[cols_all].copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)

    if len(data) < MIN_FILAS:
        print(f"{materia} (Sem {semestre}): pocos registros ({len(data)}).")
        return

    non_constant = [c for c in data.columns if data[c].nunique(dropna=False) > 1]
    data = data[non_constant]
    if COL_SEM_INT not in data.columns:
        print(f"{materia} (Sem {semestre}): no está '{COL_SEM_INT}' tras filtrar. Se omite.")
        return
    if data.shape[1] < 2:
        print(f"{materia} (Sem {semestre}): columnas insuficientes ({data.shape[1]}).")
        return

    corr_kendall = kendall_matrix(data)

    print(f"\n{materia} — Semestre {semestre} — Kendall Tau-b con '{COL_SEM_INT}':")
    print(corr_kendall[COL_SEM_INT].sort_values(ascending=False).to_string())

    n = len(corr_kendall.columns)
    fig_w = max(12, 0.7 * n)
    fig_h = max(10, 0.6 * n)

    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(corr_kendall, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, square=True, cbar=True)
    plt.title(f"Matriz Kendall Tau-b — {materia} | Sem {semestre}", fontsize=14, pad=20)
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
