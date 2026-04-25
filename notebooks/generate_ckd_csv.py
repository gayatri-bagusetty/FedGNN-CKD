# generate synthetic CKD dataset using CTGAN
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

df = pd.read_csv("../data/raw/ckd_dataset.csv")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

model = CTGANSynthesizer(metadata)
model.fit(df)

synthetic_data = model.sample(num_rows=300)

synthetic_data.to_csv("../data/raw/ckd_synthetic.csv", index=False)

print("Synthetic dataset shape:", synthetic_data.shape)

duplicates = synthetic_data.merge(df, how='inner')
print("Number of identical rows:", len(duplicates))

synthetic_df = pd.read_csv("../data/raw/ckd_synthetic.csv")

needed = 300 - len(synthetic_df)
extra_samples = model.sample(num_rows=needed)
synthetic_df = pd.concat([synthetic_df, extra_samples], ignore_index=True)
synthetic_df = synthetic_df[synthetic_df['class'].isin(['ckd','notckd'])]
synthetic_df = synthetic_df.head(300)
synthetic_df.to_csv("../data/raw/ckd_synthetic.csv", index=False)

print("Final synthetic dataset size:", synthetic_df.shape)