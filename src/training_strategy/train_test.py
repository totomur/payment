import pandas as pd
from sklearn.model_selection import train_test_split

def train_test(df_combined):
    # Saco las etiquetas del ultimo periodo, no cumplen ninguna funcion.

    ultimo_periodo_sin_etiqueta = df_combined [df_combined["PERIODO_FACTURADO"] == pd.Timestamp("2024-07-01").date()]

    df_combined.drop(df_combined[df_combined["PERIODO_FACTURADO"] == pd.Timestamp("2024-07-01").date()].index, inplace=True)

    #df_combined.fillna(0,inplace=True)

    # Asegurarse de que 'PERIODO_FACTURADO' sea de tipo datetime
    df_combined["PERIODO_FACTURADO"] = pd.to_datetime(df_combined["PERIODO_FACTURADO"])

    # Definir características y objetivo
    X = df_combined.drop(columns=["PAGO_PM_C", "PERIODO_FACTURADO"])
    y = df_combined["PAGO_PM_C"]

    # Dividir el conjunto de datos en train y test usando una fecha específica
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,  # Puedes ajustar esto según tus necesidades
        random_state=42,
        shuffle=True  # No mezclar, ya que estamos basados en tiempo
    )

    print("The dim from the train~test df : ",X_train.shape , X_test.shape)

    return  X_train, X_test, y_train, y_test