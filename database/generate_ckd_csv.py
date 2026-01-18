import pandas as pd
import numpy as np

# Define the features exactly as in your pipeline
COMMON_FEATURES = [
    'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu',
    'sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad',
    'appet','pe','ane','classification'
]

# Number of synthetic patients
num_rows = 100

# Generate random synthetic data
np.random.seed(42)
data = {
    'age': np.random.randint(18, 90, num_rows),
    'bp': np.random.randint(60, 180, num_rows),
    'sg': np.random.choice([1.005,1.010,1.015,1.020,1.025], num_rows),
    'al': np.random.choice([0,1,2,3,4,5], num_rows),
    'su': np.random.choice([0,1,2,3,4,5], num_rows),
    'rbc': np.random.choice(['normal','abnormal'], num_rows),
    'pc': np.random.choice(['normal','abnormal'], num_rows),
    'pcc': np.random.choice(['present','notpresent'], num_rows),
    'ba': np.random.choice(['present','notpresent'], num_rows),
    'bgr': np.random.randint(70, 250, num_rows),
    'bu': np.random.randint(5, 60, num_rows),
    'sc': np.round(np.random.uniform(0.5, 8.0, num_rows),2),
    'sod': np.random.randint(100, 150, num_rows),
    'pot': np.round(np.random.uniform(3.0, 8.0, num_rows),1),
    'hemo': np.round(np.random.uniform(5.0, 18.0, num_rows),1),
    'pcv': np.random.randint(15, 55, num_rows),
    'wbcc': np.random.randint(4000, 15000, num_rows),
    'rbcc': np.round(np.random.uniform(2.0, 6.0, num_rows),1),
    'htn': np.random.choice(['yes','no'], num_rows),
    'dm': np.random.choice(['yes','no'], num_rows),
    'cad': np.random.choice(['yes','no'], num_rows),
    'appet': np.random.choice(['good','poor'], num_rows),
    'pe': np.random.choice(['yes','no'], num_rows),
    'ane': np.random.choice(['yes','no'], num_rows),
    'classification': np.random.choice(['ckd','notckd'], num_rows)
}

df = pd.DataFrame(data)

# Save CSV
df.to_csv("synthetic_ckd.csv", index=False)
print("Synthetic CKD CSV generated: synthetic_ckd.csv")