import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#1. Criar dados fictícios
data = {
    'produto_id': [1,2,3,4,5],
    'vendas_ult_7_dias': [10, 2, 15, 5, 0],
    'vendas_ult_30_dias': [50, 10, 60, 20, 5],
    'estoque_atual': [5, 8, 2, 10, 1],
    'dias_para_falta': [1, 5, 0, 10, 0]  # target
}

df = pd.DataFrame(data)

df['vai_faltar']