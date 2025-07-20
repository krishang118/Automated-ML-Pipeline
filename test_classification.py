import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random
np.random.seed(42)
X, y = make_classification(n_samples=20000, n_features=6, n_informative=4,
                           n_redundant=1, n_classes=2, weights=[0.7, 0.3],
                           flip_y=0.02, class_sep=1.5, random_state=42)
df = pd.DataFrame(X, columns=['age', 'income', 'engagement_score', 'visits_per_month',
                              'time_on_site', 'interaction_rate'])
df['age'] = np.clip((df['age'] * 10 + 40).round(), 18, 70)
df['income'] = np.clip((df['income'] * 25000 + 60000).round(), 20000, 200000)
df['engagement_score'] = np.clip(df['engagement_score'] * 15 + 50, 0, 100).round(2)
df['visits_per_month'] = np.clip((df['visits_per_month'] * 2 + 5).round(), 0, 30)
df['time_on_site'] = np.clip((df['time_on_site'] * 5 + 10).round(2), 0.5, 60)
df['interaction_rate'] = np.clip((df['interaction_rate'] * 0.3 + 0.5).round(2), 0.0, 1.0)
df['location'] = np.random.choice(['Urban', 'Suburban', 'Rural'], size=len(df), p=[0.4, 0.4, 0.2])
df['device_type'] = np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=len(df), p=[0.6, 0.3, 0.1])
df['membership'] = np.random.choice(['Free', 'Silver', 'Gold', 'Platinum'], size=len(df), p=[0.5, 0.3, 0.15, 0.05])
df['target'] = np.where(y == 1, 'yes', 'no')
for col in ['visits_per_month', 'engagement_score', 'interaction_rate']:
    df.loc[df.sample(frac=0.03, random_state=42).index, col] = np.nan
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("classification_data.csv", index=False)
print("'classification_data.csv' created successfully.")
