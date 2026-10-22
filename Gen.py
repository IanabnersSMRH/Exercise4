from faker import Faker
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

fake = Faker('es_MX')
Faker.seed(42)
np.random.seed(42)

# 1. Generación programática de dataset con Faker (incluyendo campos adicionales)
registros = []
for i in range(1000):
    registros.append({
        'id_cliente': i + 1,
        'nombre': fake.name(),
        'email': fake.email(),
        'direccion': fake.address().replace('\n', ', '),  # Campo adicional 1
        'telefono': fake.phone_number(),                  # Campo adicional 2
        'estado_civil': np.random.choice(['Soltero', 'Casado', 'Divorciado'], p=[0.5, 0.4, 0.1]),
        'ingreso_anual': round(np.random.uniform(100000, 1000000), 2),
        'churn': np.random.choice([0, 1], p=[0.8, 0.2])  # 20% tasa de abandono
    })

df_sintetico = pd.DataFrame(registros)

# 2. Codificación de variables categóricas (estado_civil a columnas numéricas binarias)
df_sintetico = pd.get_dummies(df_sintetico, columns=['estado_civil'], drop_first=False)

# 3. Selección de Datos Objetivo (Target) y Características (Features)
# Excluimos identificadores, datos de texto libre y el target
X = df_sintetico.drop(columns=['id_cliente', 'nombre', 'email', 'direccion', 'telefono', 'churn'])
y = df_sintetico['churn']

# 4. Partición Estratificada (Train 80%, Test 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Dimensiones Train Set: {X_train.shape} | Proporción Churn: {y_train.mean():.2f}")
print(f"Dimensiones Test Set: {X_test.shape} | Proporción Churn: {y_test.mean():.2f}")