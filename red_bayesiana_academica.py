import pandas as pd

from pgmpy.estimators import HillClimbSearch, BDeuScore, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel

ARCHIVO = "Base_de_datos_Robusta_RedBayesiana_Bien.csv"

df = pd.read_csv(ARCHIVO, encoding="utf-8")

cols_red = [
    "PERIODO 1",
    "PERIODO 2",
    "PERIODO 3",
    "CALIFICACION",
    "ASISTENCIA PERIODO 1",
    "ASISTENCIA PERIODO 2",
    "ASISTENCIA PERIODO 3",
    "TOTAL DE ASISTENCIAS",
    "SEMESTRE DE INTERRUPCION"
]

data = df[cols_red].dropna().copy()

for c in cols_red:
    data[c] = data[c].astype(str)

print("Filas usadas para la RB:", len(data))

hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=BDeuScore(data))

print("\nEstructura aprendida (arcos):")
for edge in best_model.edges():
    print(edge)

model = BayesianModel(best_model.edges())
model.fit(data, estimator=BayesianEstimator, prior_type="BDeu", n_jobs=1)

print("\nModelo entrenado. Nodos:", model.nodes())

infer = VariableElimination(model)

evidence_ejemplo = {
    "PERIODO 1": "Alto",
    "PERIODO 2": "Alto",
    "PERIODO 3": "Alto",
    "CALIFICACION": "Alto",
}

q = infer.query(
    variables=["SEMESTRE DE INTERRUPCION"],
    evidence=evidence_ejemplo,
    show_progress=False
)

print("\nP(SEMESTRE DE INTERRUPCION | todo Bajo en calificaciones):")
print(q)
