import pandas as pd
import prince
import matplotlib.pyplot as plt

df = pd.read_csv("datos_socioeconomicos.csv")

mca = prince.MCA(n_components=2, random_state=42)
mca = mca.fit(df)
coordenadas = mca.transform(df)

plt.figure(figsize=(10, 8))
plt.scatter(coordenadas[0], coordenadas[1], alpha=0.7, color='steelblue', edgecolor='black')
plt.title('Proyección 2D - MCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.grid(True)
plt.show()