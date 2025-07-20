import pandas as pd
import numpy as np
import random
np.random.seed(42)
random.seed(42)
n_samples = 20000
noise_std = 3.0
age = np.clip(np.random.normal(35, 12, n_samples), 18, 70)
income = np.clip(np.random.lognormal(mean=11, sigma=0.5, size=n_samples), 30000, 200000)
visits = np.random.poisson(8, size=n_samples)
rating = np.clip(np.random.normal(3.8, 0.6, n_samples), 1.0, 5.0)
engagement = np.random.beta(2.2, 4.8, n_samples) * 100
location = np.random.choice(['Urban', 'Suburban', 'Rural'], size=n_samples)
product_type = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
season = np.random.choice(['Spring', 'Summer', 'Autumn', 'Winter'], size=n_samples)
location_map = {'Urban': 1.2, 'Suburban': 0.9, 'Rural': 1.0}
product_map = {'A': 1.0, 'B': 1.5, 'C': 2.0, 'D': 0.8}
season_map = {'Spring': 1.1, 'Summer': 1.3, 'Autumn': 0.9, 'Winter': 0.7}
target = (
    0.00005 * (income ** 1.1)
    - 2 * np.log1p(visits)
    + 40 * np.sin(age / 10)
    + 20 * (rating ** 2)
    + 500 * np.sqrt(engagement)
    + 0.3 * income * engagement / 1e5 
    + [location_map[loc] * 1000 for loc in location]
    + [product_map[ptype] * 1500 for ptype in product_type]
    + [season_map[s] * 700 for s in season]
    + np.random.normal(0, noise_std, n_samples)
)
df = pd.DataFrame({
    'age': np.round(age),
    'income': np.round(income),
    'visits': visits,
    'rating': rating,
    'engagement': np.round(engagement, 2),
    'location': location,
    'product_type': product_type,
    'season': season,
    'target': np.round(target, 2)
})
for col in ['rating', 'engagement', 'visits']:
    df.loc[df.sample(frac=0.03).index, col] = np.nan
df.to_csv("regression_data.csv", index=False)
print("'regression_data.csv' created successfully.")
