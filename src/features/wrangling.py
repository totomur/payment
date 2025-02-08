import pandas as pd

def wrangling (df):    
    #Elimino las columnas que estan en la posicion de 50 en adelante, son inutiles y no aportan valor, no pertenecen a lo que el segmento maneja sino a datos de orientados a B2C
    df.drop(columns=df.columns[50:], inplace = True)

    #Formateo algunas columnas relacionadas a fechas como formato manejable 
    df['PERIODO_FACTURADO'] = df['PERIODO_FACTURADO'].astype(str)
    df['PERIODO_VENCIMIENTO'] = df['PERIODO_VENCIMIENTO'].astype(str)
    df["CICLO"] = df["CICLO"].astype(str).str.replace(".0" , "",regex=True)

    # Convertir PERIODO_FACTURADO a principio de mes
    df['PERIODO_FACTURADO'] = pd.to_datetime(df['PERIODO_FACTURADO'] + '01', format='%Y%m%d') 

    # Convertir PERIODO_VENCIMIENTO a fin de mes
    df['PERIODO_VENCIMIENTO'] = pd.to_datetime(df['PERIODO_VENCIMIENTO'] + '01', format='%Y%m%d') + pd.offsets.MonthEnd(0)

    #df["CICLO"] = pd.to_datetime(df["CICLO"] , format="%Y%m%d",coerce = True)

    df["PERIODO_CONTRATO"] = (df['PERIODO_VENCIMIENTO']-df['PERIODO_FACTURADO']).dt.days


    #Formateo la columna cliente para que sea legible
    df["CLIENTE"] = df["CLIENTE"].astype(str).str.split('-').str[0]

    df['PERIODO_FACTURADO'] = df['PERIODO_FACTURADO'].dt.date
    df['PERIODO_VENCIMIENTO'] = df['PERIODO_VENCIMIENTO'].dt.date

    # Elimino las variables que no aportan dado que fueron creadas principalmente para el segmento B2C
    drop_cols = ["CICLO","CARTERA_CANALES","CARRIERS","INTRAGRUPO","SEGMENTO","SEGMENTO_AGRUPADO","RIESGO_ORIGINACION","RIESGO_CARTERA","FEC_ALTA","TIPO_DOCUMENTO","SEGMENTO_HOMOLOGADO"]
    df.drop(columns=drop_cols , inplace = True)

    # Ordeno por periodo
    df = df.sort_values(by='PERIODO_FACTURADO', ascending=True)
    periodos = df["PERIODO_FACTURADO"].unique()

    return df

