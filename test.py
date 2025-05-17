import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv('fights.csv')

# 2. Drop identifier columns
df_model = df.drop(columns=['fighter1', 'fighter2'])

# 3. Split into X (features) and y (target)
X = df_model.drop(columns=['fight_outcome'])
y = df_model['fight_outcome']

# 4. Train/test split (optional but helps sanity‐check)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Fit Random Forest
rf = RandomForestClassifier(n_estimators=200, 
                            max_depth=None, 
                            random_state=42, 
                            n_jobs=-1)
rf.fit(X_train, y_train)

# 6. Compute importances and rank
importances = rf.feature_importances_
feat_names = X.columns

feat_imp_df = (
    pd.DataFrame({'feature': feat_names, 'importance': importances})
      .sort_values('importance', ascending=False)
      .reset_index(drop=True)
)

# 7. Display
print("Feature ranking by Random Forest importance:\n")
for i, row in feat_imp_df.iterrows():
    print(f"{i+1:2d}. {row['feature']:>20s} — {row['importance']:.4f}")
