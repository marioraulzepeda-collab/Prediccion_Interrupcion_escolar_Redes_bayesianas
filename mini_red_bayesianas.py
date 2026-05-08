import pandas as pd
from pgmpy.estimators import HillClimbSearch, BDeuScore, BayesianEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

ARCHIVO = "Base_de_datos_Robusta_RedBayesiana_Bien.csv"

MIN_FILAS = 50

cols_rb = [
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

df = pd.read_csv(ARCHIVO, encoding="utf-8")

for c in ["GENERACION", "CARRERA", "SEMESTRE"] + cols_rb:
    if c not in df.columns:
        raise ValueError(f"Falta la columna '{c}' en el CSV.")

for c in cols_rb + ["GENERACION", "CARRERA", "SEMESTRE"]:
    df[c] = df[c].astype(str).str.strip()

print("Total de filas en la BD robusta:", len(df))

gens = sorted(df["GENERACION"].dropna().unique())

for gen in gens:
    df_gen = df[df["GENERACION"] == gen].copy()
    carreras = sorted(df_gen["CARRERA"].dropna().unique())

    for carr in carreras:
        df_car = df_gen[df_gen["CARRERA"] == carr].copy()
        semestres = sorted(df_car["SEMESTRE"].dropna().unique())

        for sem in semestres:
            df_block = df_car[df_car["SEMESTRE"] == sem].copy()

            data = df_block[cols_rb].dropna().copy()

            n_filas = len(data)
            titulo = f"GEN {gen} | CARRERA {carr} | SEM {sem}"

            if n_filas < MIN_FILAS:
                print(f"⏭️ {titulo}: solo {n_filas} filas, se omite (menos de {MIN_FILAS}).")
                continue

            for c in cols_rb:
                data[c] = data[c].astype(str).str.strip()

            print("\n" + "="*80)
            print(f"Entrenando RB para: {titulo}")
            print(f"Filas usadas: {n_filas}")

            hc = HillClimbSearch(data)
            best_model = hc.estimate(scoring_method=BDeuScore(data))

            print("Arcos aprendidos (estructura):")
            for edge in best_model.edges():
                print("  ", edge)

            model = BayesianModel(best_model.edges())
            model.fit(data, estimator=BayesianEstimator, prior_type="BDeu", n_jobs=1)

            print("Nodos del modelo:", list(model.nodes()))

            infer = VariableElimination(model)

            try:
                evidence_ejemplo = {
                    "PERIODO 1": "Bajo",
                    "PERIODO 2": "Bajo",
                    "PERIODO 3": "Bajo",
                    "CALIFICACION": "Bajo"
                }

                q = infer.query(
                    variables=["SEMESTRE DE INTERRUPCION"],
                    evidence=evidence_ejemplo,
                    show_progress=False
                )

                print("\nP(SEMESTRE DE INTERRUPCION | todo 'Bajo' en calificaciones):")
                print(q)
            except Exception as e:
                print("No se pudo hacer la inferencia ejemplo en este bloque:", e)
