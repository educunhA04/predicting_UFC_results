import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
def load_data():
    df = pd.read_csv('data/ufc_fights_cleaned.csv')
    fighters = pd.concat([df['fighter1'], df['fighter2']]).unique()
    return df, sorted(fighters)

# Train the SVM model
def train_model(df):
    diff_features = []
    for feature in ['Weight', 'Reach', 'SLpM', 'StrAcc', 'SApM', 'StrDef', 
                    'TDAvg', 'TDAcc', 'TDDef', 'SubAvg', 'Wins', 'Losses', 'Draws', 'Height_in']:
        df[f'diff_{feature}'] = df[f'fighter1_{feature}'] - df[f'fighter2_{feature}']
        diff_features.append(f'diff_{feature}')

    stances = ['Open stance', 'Orthodox', 'Southpaw', 'Switch']
    for stance in stances:
        df[f'diff_Stance_{stance}'] = (
            df[f'fighter1_Stance_{stance}'].astype(int) -
            df[f'fighter2_Stance_{stance}'].astype(int)
        )
    X = df[diff_features]
    y = df['fight_outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svm = SVC(C=1, gamma='scale', kernel='rbf', probability=True, random_state=42)
    svm.fit(X_scaled, y)
    
    return svm, scaler, diff_features

# Fighter comparison function
def compare_fighters(df, model, scaler, feature_columns, fighter1, fighter2):
    fighter1_stats = df[df['fighter1'] == fighter1].iloc[0] if fighter1 in df['fighter1'].values else df[df['fighter2'] == fighter1].iloc[0]
    fighter2_stats = df[df['fighter1'] == fighter2].iloc[0] if fighter2 in df['fighter1'].values else df[df['fighter2'] == fighter2].iloc[0]
    
    if fighter1_stats['fighter1'] == fighter1 and fighter2_stats['fighter1'] == fighter2:
        pass
    elif fighter1_stats['fighter2'] == fighter1 and fighter2_stats['fighter2'] == fighter2:
        pass
    else:
        if fighter1_stats['fighter2'] == fighter1:
            fighter1_stats, fighter2_stats = fighter2_stats, fighter1_stats
    
    features = {}
    for feature in ['Weight', 'Reach', 'SLpM', 'StrAcc', 'SApM', 'StrDef', 
                    'TDAvg', 'TDAcc', 'TDDef', 'SubAvg', 'Wins', 'Losses', 'Draws', 'Height_in']:
        features[f'diff_{feature}'] = fighter1_stats[f'fighter1_{feature}'] - fighter2_stats[f'fighter2_{feature}']
    
    for stance in ['Open stance', 'Orthodox', 'Southpaw', 'Switch']:
        features[f'diff_Stance_{stance}'] = int(fighter1_stats[f'fighter1_Stance_{stance}']) - int(fighter2_stats[f'fighter2_Stance_{stance}'])
    
    feature_df = pd.DataFrame([features])[feature_columns]
    scaled_features = scaler.transform(feature_df)
    
    prediction = model.predict(scaled_features)[0]
    proba = model.predict_proba(scaled_features)[0]
    
    fighter1_win_prob = proba[1] if prediction == 1 else proba[0]
    fighter2_win_prob = 1 - fighter1_win_prob
    
    stats_to_show = [
        'Weight', 'Reach', 'Height_in', 'Wins', 'Losses', 'Draws',
        'SLpM', 'StrAcc', 'StrDef', 'TDAvg', 'TDAcc', 'TDDef', 'SubAvg'
    ]
    
    fighter1_display = {stat: fighter1_stats[f'fighter1_{stat}'] for stat in stats_to_show}
    fighter2_display = {stat: fighter2_stats[f'fighter2_{stat}'] for stat in stats_to_show}
    
    stances = ['Open stance', 'Orthodox', 'Southpaw', 'Switch']
    fighter1_stance = next((s for s in stances if fighter1_stats[f'fighter1_Stance_{s}']), 'Unknown')
    fighter2_stance = next((s for s in stances if fighter2_stats[f'fighter2_Stance_{s}']), 'Unknown')
    
    return {
        'prediction': prediction,
        'fighter1_prob': fighter1_win_prob,
        'fighter2_prob': fighter2_win_prob,
        'fighter1_stats': fighter1_display,
        'fighter2_stats': fighter2_display,
        'fighter1_stance': fighter1_stance,
        'fighter2_stance': fighter2_stance,
        'fighter1_age': fighter1_stats['fighter1_Age'],
        'fighter2_age': fighter2_stats['fighter2_Age']
    }

# Function to create comparison charts
def create_comparison_charts(fighter1, fighter2, fighter1_stats, fighter2_stats):
    # Select key metrics for visualization
    key_metrics = ['SLpM', 'StrAcc', 'StrDef', 'TDAvg', 'TDAcc', 'TDDef', 'SubAvg']
    
    # Create radar chart
    categories = key_metrics
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Fighter 1 values
    values1 = [fighter1_stats[metric] for metric in key_metrics]
    values1 += values1[:1]
    ax.plot(angles, values1, linewidth=1, linestyle='solid', label=fighter1, color='#FF4B4B')
    ax.fill(angles, values1, alpha=0.25, color='#FF4B4B')
    
    # Fighter 2 values
    values2 = [fighter2_stats[metric] for metric in key_metrics]
    values2 += values2[:1]
    ax.plot(angles, values2, linewidth=1, linestyle='solid', label=fighter2, color='#1F77B4')
    ax.fill(angles, values2, alpha=0.25, color='#1F77B4')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Fighting Skills Comparison', size=20, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.close(fig)
    
    # Create bar chart for physical attributes
    physical_metrics = ['Weight', 'Reach', 'Height_in']
    f1_physical = [fighter1_stats[metric] for metric in physical_metrics]
    f2_physical = [fighter2_stats[metric] for metric in physical_metrics]
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(physical_metrics))
    width = 0.35
    
    rects1 = ax2.bar(x - width/2, f1_physical, width, label=fighter1, color='#FF4B4B')
    rects2 = ax2.bar(x + width/2, f2_physical, width, label=fighter2, color='#1F77B4')
    
    ax2.set_ylabel('Value')
    ax2.set_title('Physical Attributes Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(physical_metrics)
    ax2.legend()
    
    plt.close(fig2)
    
    return fig, fig2