import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ARCHIVO = r"datos_socioeconomicos.csv"

soc_cols = [
    "Edad",
    "Sexo",
	"Situación sentimental",
    "Personas en casa",
    "Tutor trabaja",
    "Servicios básicos",
    "Casa propia",
    "Padres con prepa",
    "Transporte",
    "Aporta al hogar",
    "Puede estudiar en casa",
    "Dispositivos",
    "Sabe otro idioma",	
    "Tiempo a la escuela",
    "Viaja en vacaciones"

]

try:
    df = pd.read_csv(ARCHIVO, encoding="utf-8")
except:
    df = pd.read_csv(ARCHIVO, encoding="utf-8-sig")

data = df[soc_cols].copy()

for c in soc_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

data = data.fillna(0)

corr = data.corr(method="spearman").fillna(0)

n = len(corr.columns)
plt.figure(figsize=(max(12, 0.65*n), max(10, 0.60*n)))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=.5, square=True, cbar=True)

plt.title("Matriz de correlación de Spearman — Encuesta socioeconómica", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
