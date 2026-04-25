import pandas as pd
df = pd.read_csv("../data/raw/ckd_synthetic.csv")
# df = df[df['class'].isin(['ckd','notckd'])]
val = df['class'].value_counts()
print(val)