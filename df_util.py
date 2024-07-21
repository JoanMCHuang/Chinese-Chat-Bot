import pandas as pd

def create_DataFrame(doc, fields):
    columns = []
    for _ in fields:
        columns.append([])
    
    for token in doc:
#         print(dir(token))
        for i, name in enumerate(fields):            
            columns[i].append(token.__getattribute__(name))
    df = pd.DataFrame()
    for i, name in enumerate(fields):
        df[name] = columns[i]
    return df