#Input paths

#path_1 = r"/home/tomas-mur/Documentos/Codigos/payment/datasets/raw/Agosto/Detalle_Ratio_1079_20240801151755.txt" #Davox P1
#path_2 = r"/home/tomas-mur/Documentos/Codigos/payment/datasets/raw/Agosto/Detalle_Ratio_1080_20240802120040.txt" #Davox P2
#path_3 = r"/home/tomas-mur/Documentos/Codigos/payment/datasets/raw/Agosto/Detalle_Ratio_1081_20240802135513.txt" #FS FIJA P1
#path_4 = r"/home/tomas-mur/Documentos/Codigos/payment/datasets/raw/Agosto/Detalle_Ratio_1082_20240805111624.txt" #FS FIJA P2
#path_5 = r"/home/tomas-mur/Documentos/Codigos/payment/datasets/raw/Agosto/Detalle_Ratio_1083_20240805101000.txt" #FS MOVIL P1
#path_6 = r"/home/tomas-mur/Documentos/Codigos/payment/datasets/raw/Agosto/Detalle_Ratio_1084_20240805125725.txt" #SCL P1
#path_7 = r"/home/tomas-mur/Documentos/Codigos/payment/datasets/raw/Agosto/Detalle_Ratio_1086_20240805153333.txt" #SCL P2

path_1 = r"~/buckets/b1/datasets/Detalle_Ratio_1079_20240801151755.txt" #Davox P1
path_2 = r"~/buckets/b1/datasets/Detalle_Ratio_1080_20240802120040.txt" #Davox P2
path_3 = r"~/buckets/b1/datasets/Detalle_Ratio_1081_20240802135513.txt" #FS FIJA P1
path_4 = r"~/buckets/b1/datasets/Detalle_Ratio_1082_20240805111624.txt" #FS FIJA P2
path_5 = r"~/buckets/b1/datasets/Detalle_Ratio_1083_20240805101000.txt" #FS MOVIL P1
path_6 = r"~/buckets/b1/datasets/Detalle_Ratio_1084_20240805125725.txt" #SCL P1
path_7 = r"~/buckets/b1/datasets/Detalle_Ratio_1086_20240805153333.txt" #SCL P2
#Reading each dataset

df_1 = pd.read_csv(path_1 , sep = "|" )
df_2 = pd.read_csv(path_2 , sep = "|" )
df_3 = pd.read_csv(path_3 , sep = "|" )
df_4 = pd.read_csv(path_4 , sep = "|" )
df_5 = pd.read_csv(path_5 , sep = "|" )
df_6 = pd.read_csv(path_6 , sep = "|" )
df_7 = pd.read_csv(path_7 , sep = "|" )

#Joining to get the final dataset

df  = pd.concat([df_1 , df_2 , df_3 , df_4 , df_5 , df_6 , df_7])

#Releasing some memory

del df_1
del df_2
del df_3
del df_4
del df_5
del df_6
del df_7 