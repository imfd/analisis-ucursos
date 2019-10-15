import sqlalchemy
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from content_utils import *

engine = sqlalchemy.create_engine('mysql+mysqlconnector://root@localhost/ucursos')

mensajes_query = """
    select 
        men_id as id_mensaje, 
        men_rut as autor,
        men_contenido as contenido,
        men_fecha_creacion as fecha_creacion
    from mensajes_ingenieria
    where men_censurado = 0;
"""

temas_df = pd.read_sql("SELECT * FROM temas;", engine)
mensajes_df = pd.read_sql(mensajes_query, engine)
mensajes_df = mensajes_df.set_index("id_mensaje")
respuestas_df = pd.read_sql("SELECT id_mensaje, raiz FROM respuestas", engine)
respuestas_df = respuestas_df.set_index("id_mensaje")

mensajes2_df = respuestas_df.join(mensajes_df)
mensajes2_df['id_mensaje'] = mensajes2_df.index
mensajes2_df = mensajes2_df.set_index("raiz")
mensajes2_df = mensajes2_df.rename(columns={'fecha_creacion': 'fecha_creacion_mensaje'})

df = mensajes2_df.join(temas_df)
df['raiz'] = df.index
df = df.set_index("id_mensaje")


###### crear documentos
# 1 doc = concat mensajes de un tema

documents = defaultdict(list)

for _, row in tqdm(df.iterrows(), total=len(df.index)):
    raiz = row["raiz"]
    documents[raiz].append(row["contenido"])


#### tokenize documents

doc_tokens = defaultdict(list)

for raiz, docs in tqdm(documents.items()):
    for doc in docs:
        doc_tokens[raiz].append(tokenize(doc, nlp))

import pickle

with open('data/doc_tokens.pkl', 'wb') as f:
    pickle.dump(doc_tokens, f)