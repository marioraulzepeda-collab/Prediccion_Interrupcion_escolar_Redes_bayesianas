import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ARCHIVO = r"BD_especial_Gen2022_2025.csv"
MATERIA_OBJETIVO = "ECOLOGÍA"
MIN_FILAS = 5

cal_cols  = ["PARCIAL 1","PARCIAL 2","PARCIAL 3","CALIFICACION"]
asis_cols = ["ASISTENCIAS 1","ASISTENCIAS 2","ASISTENCIAS 3","TOTAL ASISTENCIAS"]
soc_cols  = ["Edad","Sexo","Sit_Sentimental","Personas_Hogar","Tutores_trabajan",
             "Servicios","Casa_propia","Concluir_Prepa","Transporte_Escuela",
             "Contribuir_Eco","Rea_Tareas_Hogar","Disp_Elec","Idioma",
             "Tiempo_Casa_Escuela","Viajar"]

try:
    df = pd.read_csv(ARCHIVO, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(ARCHIVO, encoding="utf-8-sig")

df["NOMBRE ASIGNATURA"] = df["NOMBRE ASIGNATURA"].astype(str).str.strip().str.upper()

df["SEM_BIN"] = df["SEMESTRE DE INTERRUPCION"].apply(lambda x: 0 if x == 6 else 1 if x in [1,2,3,4,5] else 0)

cols_all = cal_cols + asis_cols + soc_cols + ["SEM_BIN"]
for c in cols_all:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

if MATERIA_OBJETIVO.strip() == "":
    materias = sorted(df["NOMBRE ASIGNATURA"].dropna().unique())
else:
    materias = [MATERIA_OBJETIVO.strip().upper()]

print("\nMaterias que se graficarán:")
for i, mat in enumerate(materias, 1):
    print(f"{i:>2}. {mat}")

def grafica_materia(df_mat, materia):
    data = df_mat[cols_all].copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)

    if len(data) < MIN_FILAS:
        print(f"{materia}: pocos registros ({len(data)}).")
        return

    corr = data.corr(method="spearman").fillna(0)

    for i in range(len(corr)):
        corr.iat[i, i] = 1.0

    print(f"\n{materia} — correlación con SEM_BIN:")
    print(corr["SEM_BIN"].sort_values(ascending=False).to_string())

    n_cols = len(corr.columns)
    fig_width = max(12, n_cols * 0.7)
    fig_height = max(10, n_cols * 0.6)

    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, square=True, cbar=True)

    plt.title(f"Matriz Spearman — {materia}", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout(pad=3.0)
    plt.show()

for materia in materias:
    df_materia = df[df["NOMBRE ASIGNATURA"] == materia].copy()
    if df_materia.empty:
        print(f"Materia no encontrada: {materia}")
        continue
    grafica_materia(df_materia, materia)
