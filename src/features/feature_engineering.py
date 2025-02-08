import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def feature_engineering(df):
   # Creacion de columna de desplazamiento para ver el valor anterior(1 y 2 meses) de S0.

    df['S0_PREV'] = df.groupby(["CLIENTE"])['S0'].shift(1)
    df['S0_PREV2'] = df.groupby(["CLIENTE"])['S0'].shift(2)
    df['S0_PREV3'] = df.groupby(["CLIENTE"])['S0'].shift(3)

    # Diferencial entre pagos previos, si crece o disminuyen
    df['GAP_PM2M'] = df['S0_PREV'] - df['S0_PREV2']
    df['GAP_PM2M2'] = df['S0_PREV2'] - df['S0_PREV3']
    #df['GAP_PM2M'] = df['GAP_PM2M'].fillna(0)
    #df['GAP_PM2M2'] = df['GAP_PM2M2'].fillna(0)

    # Generacion de  media movil de 3 meses para la facturacion / pagos a S0

    #df["S0_3EMA"] = df.groupby(["CLIENTE"])['FACTURA_REAL'].rolling(window=3).mean()
    #df["S0_3EMA"] = df.groupby(["CLIENTE"])['S0'].rolling(window=3).mean()

    df["FACT_3EMA"] = df.groupby("CLIENTE")['FACTURA_REAL'].transform(lambda x: x.rolling(window=3).mean())
    df["S0_3EMA"] = df.groupby("CLIENTE")['S0'].transform(lambda x: x.rolling(window=3).mean())

    # Determinacion si un cliente pago el mes anterior en S0
    df['PAGO_UM'] = df.apply(lambda row: 1 if row['S0_PREV'] > 0  else 0, axis=1)

    # Determinar si un cliente pagó el mes anterior y no pagó este mes
    df['PAGO_UM_C'] = df.apply(lambda row: 1 if row['S0_PREV'] > 0 and row['S0'] == 0 else 0, axis=1)



    # Dropeo la columna
    #df = df.drop(columns="S0_PREV") lo comente el 22-08-2024 a modo de prueba

    # Generacion de la variable objetivo, es decir, a predecir

    # Creacion de columna de desplazamiento para ver el valor futuro de S0
    df['S0_NEXT'] = df.groupby('CLIENTE')['S0'].shift(-2)

    # Determinacion de si un cliente pagará el próximo mes en la semana 0
    df['PAGO_PM_C'] = df['S0_NEXT'].apply(lambda x: 1 if x > 0 else 0)

    # Drop de la columna temporal S0_NEXT
    df = df.drop(columns=['S0_NEXT']) 

    # Agrupo los pagos cada 4 semanas, a excepcion de pago a vencimiento (S0).

    df["0S"] = (df["S0"]  ) 
    df["4S"] = ( df["S1"] +df["S2"] +df["S3"] + df["S4"]  ) 
    df["8S"] = (df["S5"] + df["S6"] +df["S7"] +df["S8"] ) 
    df["12S"] = (df["S9"] + df["S10"] +df["S11"] +df["S12"] ) 
    df["16S"] = (df["S13"] + df["S14"] +df["S15"] +df["S16"] ) 
    df["20S"] = (df["S17"] + df["S18"] +df["S19"] +df["S20"] ) 
    df["24S"] = (df["S21"] + df["S22"] +df["S23"] +df["S24"] ) 

    # Genero una variable que sea proporcional al pago

    PR_S0 = pd.DataFrame((df.groupby("CLIENTE")["S0"].sum() / df.groupby("CLIENTE")["FACTURA_REAL"].sum()), columns=["PR_S0"])

    # saco las features que ya estan incluidas en las agrupaciones

    drop_cols_S = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10',
           'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
           'S21', 'S22', 'S23', 'S24']
    df.drop(columns=drop_cols_S,inplace=True)

    # Seteo a  Cliente como indice para facilitar el merge.

    df = df.reset_index().set_index("CLIENTE")


    # Agrupo segun antiguedad de clietne.

    df["VIGENCIA"] = np.where (df["VIGENCIA"] == "Alta Antigua Mayor a 12 meses" , "Mayor a 12 meses" , df["VIGENCIA"])
    df["VIGENCIA"] = np.where ( (df["VIGENCIA"] == "Alta Nueva hasta 3 meses") | (df["VIGENCIA"] == "Alta Nueva de 4 a 6 meses")  , "Menor o igual a 6 meses" , df["VIGENCIA"])
    df["VIGENCIA"] = np.where ( (df["VIGENCIA"] == "Entre 6 y 12 meses")  , "Alta Nueva de 7 a 12 meses" , df["VIGENCIA"])

    # soluciono inconsistencias en los segmentos de atencion

    df["SEGMENTO_ATENCION"] = np.where (df["SEGMENTO_ATENCION"] == "EMPRESAS ","EMPRESAS",df["SEGMENTO_ATENCION"]  )
    df["SEGMENTO_ATENCION"] = np.where (df["SEGMENTO_ATENCION"] == "Empresas","EMPRESAS",df["SEGMENTO_ATENCION"]  )

    # Unir los DataFrames utilizando el índice.

    df_combined = pd.merge(df, PR_S0, left_index=True, right_index=True, how='left')

    # Genero una nueva base.

    df_combined = df_combined.reset_index()
    df_combined = df_combined.sort_values("PERIODO_VENCIMIENTO")

    # Saco las columnas que no van a aportar para el train test validation
    drop_comb = ["PREFIJO" , "index" ,"NUMERO_FACTURA"  , "PERIODO_VENCIMIENTO" , "DEPARTAMENTO", "NUM_IDENT","CLIENTE","OFERTA" ]
    df_combined.drop(columns=drop_comb, inplace=True)


    # Aplico ordinal encoder para el atributo "VIGENCIA" que representa la permanencia dentro de TEF de c/empresa

    # Lista de categorías ordenadas de menor a mayor

    orden_categorias = ["Menor o igual a 6 meses", "Alta Nueva de 7 a 12 meses", "Mayor a 12 meses"]

    #Aplico conversion

    ord_enc = OrdinalEncoder(categories=[orden_categorias])

    # Aplicar el encoder a la columna

    df_combined['VIGENCIA'] = ord_enc.fit_transform(df_combined[['VIGENCIA']])

    ##Segmento

    # Initialize the encoder
    oh_enc_seg = OneHotEncoder(drop='first', sparse_output=False)

    # Perform the one-hot encoding
    encoded_columns_segmento = oh_enc_seg.fit_transform(df_combined[['SEGMENTO_ATENCION']])

    # Create a DataFrame with the encoded columns

    encoded_df = pd.DataFrame(encoded_columns_segmento, columns=oh_enc_seg.get_feature_names_out(['SEGMENTO_ATENCION']))

    # Concatenate the original DataFrame (excluding the original column) with the encoded DataFrame

    df_combined = pd.concat([df_combined.drop(columns=['SEGMENTO_ATENCION']), encoded_df], axis=1)

    ##Base

    oh_enc_base = OneHotEncoder(drop='first', sparse_output=False)
    encoded_columns_base = oh_enc_base.fit_transform(df_combined[["BASE"]])
    encoded_df = pd.DataFrame(encoded_columns_base , columns = oh_enc_base.get_feature_names_out(["BASE"]))
    df_combined = pd.concat( [df_combined.drop(columns=["BASE"]) , encoded_df] , axis = 1)

    df_combined["PERIODO_FACTURADO"].unique()

    return df_combined