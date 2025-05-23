import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os

# Set page config
st.set_page_config(
    page_title="UFC Fight Outcome Predictor",
    page_icon="🥊",
    layout="wide"
)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('data/ufc_fights_cleaned.csv')

    # Get unique fighters from both columns
    fighters = pd.concat([df['fighter1'], df['fighter2']]).unique()
    return df, sorted(fighters)

df, all_fighters = load_data()

# Train the SVM model (as per conclusions.md)
def train_model(df):
    # Create differential features as in test.py
    diff_features = []
    for feature in ['Weight', 'Reach', 'SLpM', 'StrAcc', 'SApM', 'StrDef', 
                    'TDAvg', 'TDAcc', 'TDDef', 'SubAvg', 'Wins', 'Losses', 'Draws', 'Height_in']:
        df[f'diff_{feature}'] = df[f'fighter1_{feature}'] - df[f'fighter2_{feature}']
        diff_features.append(f'diff_{feature}')

    # Add stance differences
    stances = ['Open stance', 'Orthodox', 'Southpaw', 'Switch']
    for stance in stances:
        df[f'diff_Stance_{stance}'] = (
            df[f'fighter1_Stance_{stance}'].astype(int) -
            df[f'fighter2_Stance_{stance}'].astype(int)
        )
    X = df[diff_features]
    y = df['fight_outcome']  # 1 if fighter1 wins, 0 if fighter2 wins
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train SVM with best parameters from conclusions
    svm = SVC(C=1, gamma='scale', kernel='rbf', probability=True, random_state=42)
    svm.fit(X_scaled, y)
    
    return svm, scaler, diff_features

model, scaler, feature_columns = train_model(df)

# Fighter comparison function
def compare_fighters(fighter1, fighter2):
    # Get stats for both fighters
    fighter1_stats = df[df['fighter1'] == fighter1].iloc[0] if fighter1 in df['fighter1'].values else df[df['fighter2'] == fighter1].iloc[0]
    fighter2_stats = df[df['fighter1'] == fighter2].iloc[0] if fighter2 in df['fighter1'].values else df[df['fighter2'] == fighter2].iloc[0]
    
    # Determine if we need to swap stats (if both were found as fighter1 or both as fighter2)
    if fighter1_stats['fighter1'] == fighter1 and fighter2_stats['fighter1'] == fighter2:
        pass
    elif fighter1_stats['fighter2'] == fighter1 and fighter2_stats['fighter2'] == fighter2:
        pass
    else:
        if fighter1_stats['fighter2'] == fighter1:
            fighter1_stats, fighter2_stats = fighter2_stats, fighter1_stats
    
    # Create feature vector for prediction
    features = {}
    for feature in ['Weight', 'Reach', 'SLpM', 'StrAcc', 'SApM', 'StrDef', 
                    'TDAvg', 'TDAcc', 'TDDef', 'SubAvg', 'Wins', 'Losses', 'Draws', 'Height_in']:
        features[f'diff_{feature}'] = fighter1_stats[f'fighter1_{feature}'] - fighter2_stats[f'fighter2_{feature}']
    
    for stance in ['Open stance', 'Orthodox', 'Southpaw', 'Switch']:
        features[f'diff_Stance_{stance}'] = int(fighter1_stats[f'fighter1_Stance_{stance}']) - int(fighter2_stats[f'fighter2_Stance_{stance}'])
    
    # Convert to dataframe for scaling
    feature_df = pd.DataFrame([features])[feature_columns]
    
    # Scale features
    scaled_features = scaler.transform(feature_df)
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    proba = model.predict_proba(scaled_features)[0]
    
    # Get win probability for fighter1
    fighter1_win_prob = proba[1] if prediction == 1 else proba[0]
    fighter2_win_prob = 1 - fighter1_win_prob
    
    # Prepare stats for display
    stats_to_show = [
        'Weight', 'Reach', 'Height_in', 'Wins', 'Losses', 'Draws',
        'SLpM', 'StrAcc', 'StrDef', 'TDAvg', 'TDAcc', 'TDDef', 'SubAvg'
    ]
    
    fighter1_display = {stat: fighter1_stats[f'fighter1_{stat}'] for stat in stats_to_show}
    fighter2_display = {stat: fighter2_stats[f'fighter2_{stat}'] for stat in stats_to_show}
    
    # Get stances
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

# Streamlit UI
st.title("🥊 UFC Fight Outcome Predictor")
st.markdown("""
This app predicts the outcome of a UFC fight between two selected fighters using a Support Vector Machine (SVM) model 
with 83.5% accuracy, based on the fighter statistics and historical fight data.
""")

st.markdown("---")

# Fighter selection
col1, col2 = st.columns(2)
with col1:
    fighter1 = st.selectbox("Select Fighter 1:", all_fighters, index=0)
with col2:
    # Ensure fighter2 is different from fighter1
    fighter2_options = [f for f in all_fighters if f != fighter1]
    fighter2 = st.selectbox("Select Fighter 2:", fighter2_options, index=0 if len(fighter2_options) > 0 else None)

if st.button("Predict Fight Outcome"):
    if fighter1 == fighter2:
        st.error("Please select two different fighters.")
    else:
        try:
            result = compare_fighters(fighter1, fighter2)
            
            st.markdown("---")
            st.header("🏆 Prediction Results")
            
            # Display prediction
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(fighter1)
                st.metric("Win Probability", f"{result['fighter1_prob']*100:.1f}%")
            with col2:
                st.subheader(fighter2)
                st.metric("Win Probability", f"{result['fighter2_prob']*100:.1f}%")
            
            # Display predicted winner
            predicted_winner = fighter1 if result['prediction'] == 1 else fighter2
            confidence = max(result['fighter1_prob'], result['fighter2_prob'])
            st.success(f"**Predicted Winner:** {predicted_winner} ({(confidence*100):.1f}% confidence)")
            
            st.markdown("---")
            st.header("📊 Fighter Statistics Comparison")
            
            # Stats comparison
            stats = result['fighter1_stats'].keys()
            stats_cols = st.columns(4)
            
            # First row - headers
            for i, col in enumerate(stats_cols):
                with col:
                    if i == 0:
                        st.markdown("**Statistic**")
                    elif i == 1:
                        st.markdown(f"**{fighter1}**")
                    elif i == 2:
                        st.markdown(f"**{fighter2}**")
                    elif i == 3:
                        st.markdown("**Difference**")
            
            # Display each stat
            for stat in stats:
                f1_val = result['fighter1_stats'][stat]
                f2_val = result['fighter2_stats'][stat]
                diff = f1_val - f2_val
                
                for i, col in enumerate(stats_cols):
                    with col:
                        if i == 0:
                            st.markdown(stat)
                        elif i == 1:
                            st.markdown(f"{f1_val:.2f}")
                        elif i == 2:
                            st.markdown(f"{f2_val:.2f}")
                        elif i == 3:
                            color = "green" if diff > 0 else "red" if diff < 0 else "gray"
                            st.markdown(f"<span style='color: {color}'>{diff:+.2f}</span>", unsafe_allow_html=True)
            
            # Additional info
            st.markdown("---")
            st.subheader("Additional Information")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{fighter1}**")
                st.markdown(f"- **Stance:** {result['fighter1_stance']}")
                st.markdown(f"- **Age:** {result['fighter1_age']:.1f} years")
            with col2:
                st.markdown(f"**{fighter2}**")
                st.markdown(f"- **Stance:** {result['fighter2_stance']}")
                st.markdown(f"- **Age:** {result['fighter2_age']:.1f} years")
            
            # Key metrics explanation
            st.markdown("""
            **Key Metrics Explanation:**
            - **SLpM:** Significant Strikes Landed per Minute
            - **StrAcc:** Striking Accuracy (%)
            - **StrDef:** Striking Defense (%)
            - **TDAvg:** Average Takedowns per 15 minutes
            - **TDAcc:** Takedown Accuracy (%)
            - **TDDef:** Takedown Defense (%)
            - **SubAvg:** Average Submissions Attempted per 15 minutes
            """)
            
        except Exception as e:
            st.error(f"Error processing fighters: {str(e)}")
            st.info("Please try different fighters. Some fighter data may be incomplete.")

st.markdown("---")
st.markdown("""
**Note:** This prediction is based on historical data and statistical modeling (SVM with 83.5% accuracy). 
Actual fight outcomes may vary due to factors not captured in the data.
""")