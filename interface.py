import streamlit as st
import matplotlib.pyplot as plt
import os
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from prediction import load_data, create_comparison_charts

# Set page config
st.set_page_config(
    page_title="UFC Fight Prediction - Complete Analysis",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (CORES ARRANJADAS!)
st.markdown("""
<style>
    .header-style {
        font-size: 24px;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: #212529;
    }
    .fighter-card {
        background-color: #FFFFFF;
        border: 1px solid #DEE2E6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        color: #212529;
    }
    .stat-diff-positive {
        color: #28A745;
        font-weight: bold;
    }
    .stat-diff-negative {
        color: #DC3545;
        font-weight: bold;
    }
    .stat-diff-neutral {
        color: #6C757D;
    }
    .prediction-banner {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.15);
    }
    .best-model-banner {
        background: linear-gradient(90deg, #4CAF50 0%, #66BB6A 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.15);
    }
    .info-box {
        background-color: #F8F9FA;
        border-left: 4px solid #007BFF;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

# Load data and train RANDOM FOREST model (the best one!)
@st.cache_data
def load_and_train_model():
    try:
        df = pd.read_csv('data/ufc_fights_cleaned.csv')
        fighters = pd.concat([df['fighter1'], df['fighter2']]).unique()
        
        # Feature engineering
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
            diff_features.append(f'diff_Stance_{stance}')

        X = df[diff_features]
        y = df['fight_outcome']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use RANDOM FOREST (the best model!) instead of SVM
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=20, 
            min_samples_split=2, 
            random_state=42
        )
        model.fit(X_scaled, y)
        
        return df, sorted(fighters), model, scaler, diff_features
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

# Function to display image if it exists
def display_image(image_path, caption="", width=None):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=caption, width=width)
    else:
        st.warning(f"Image not found: {image_path}")

# Sidebar for navigation
st.sidebar.title("🥊 UFC Analysis Hub")
analysis_section = st.sidebar.selectbox(
    "Select Section:",
    ["🎯 Fight Predictions", "📊 Model Analysis", "🏆 Key Findings", "📈 Performance Comparison", "🔍 Feature Analysis"]
)

# Main title
st.title("🥊 UFC Fight Outcome Prediction - Complete Analysis")
st.markdown(f'<div class="best-model-banner">'
                f'<h2 style="color:white;text-align:center;">🏆 NOW USING RANDOM FOREST - 87.5% ACCURACY (BEST MODEL!)</h2>'
                f'</div>', unsafe_allow_html=True)

# Load model
df, all_fighters, model, scaler, feature_columns = load_and_train_model()

if df is None:
    st.error("Failed to load data or train model. Please check your data files.")
    st.stop()

# Fighter comparison function (updated for Random Forest)
def compare_fighters_rf(df, model, scaler, feature_columns, fighter1, fighter2):
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

# Main content based on selected section
if analysis_section == "🎯 Fight Predictions":
    st.header("🎯 Fight Predictions")
    st.markdown("""
    <div class="info-box">
        <p style="font-size:16px;">Predict UFC fight outcomes using our <strong>Random Forest model with 87.5% accuracy</strong> 
        (upgraded from SVM 83.5%)! Based on comprehensive fighter statistics and historical data.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Fighter selection
    st.subheader("Select Fighters")
    col1, col2 = st.columns(2)
    with col1:
        fighter1 = st.selectbox("Fighter 1:", all_fighters, index=0, key='fighter1')
    with col2:
        fighter2_options = [f for f in all_fighters if f != fighter1]
        fighter2 = st.selectbox("Fighter 2:", fighter2_options, index=0 if len(fighter2_options) > 0 else None, key='fighter2')

    if st.button("🥊 Predict Fight Outcome", type="primary"):
        if fighter1 == fighter2:
            st.error("Please select two different fighters.")
        else:
            try:
                result = compare_fighters_rf(df, model, scaler, feature_columns, fighter1, fighter2)
                
                st.markdown("---")
                st.markdown(f'<div class="header-style">🏆 Prediction Results (Random Forest - 87.5% Accuracy)</div>', unsafe_allow_html=True)
                
                # Display prediction in cards
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<div class="metric-card"><h3 style="color:#212529;">{fighter1}</h3>', unsafe_allow_html=True)
                    st.metric("Win Probability", f"{result['fighter1_prob']*100:.1f}%", delta=f"+{(result['fighter1_prob']-0.5)*100:.1f}% vs average")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h3 style="color:#212529;">{fighter2}</h3>', unsafe_allow_html=True)
                    st.metric("Win Probability", f"{result['fighter2_prob']*100:.1f}%", delta=f"+{(result['fighter2_prob']-0.5)*100:.1f}% vs average")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display predicted winner banner
                predicted_winner = fighter1 if result['prediction'] == 1 else fighter2
                confidence = max(result['fighter1_prob'], result['fighter2_prob'])
                st.markdown(
                    f'<div class="prediction-banner">'
                    f'<h2 style="color:white;text-align:center;">🏆 PREDICTED WINNER: {predicted_winner.upper()} ({(confidence*100):.1f}% confidence)</h2>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                # Visualizations
                st.markdown("---")
                st.markdown(f'<div class="header-style">📊 Fighter Comparison Charts</div>', unsafe_allow_html=True)
                
                radar_fig, bar_fig = create_comparison_charts(fighter1, fighter2, result['fighter1_stats'], result['fighter2_stats'])
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(radar_fig)
                with col2:
                    st.pyplot(bar_fig)
                
            except Exception as e:
                st.error(f"Error processing fighters: {str(e)}")

elif analysis_section == "📊 Model Analysis":
    st.header("📊 Complete Model Analysis")
    
    # Check if results exist
    if os.path.exists('analysis/results'):
        st.subheader("📈 Class Distribution")
        display_image('analysis/results/eda/class_distribution.png', "Distribution of fight outcomes")
        
        st.subheader("🔗 Feature Correlations")
        display_image('analysis/results/eda/correlation_matrix.png', "Correlation matrix showing feature relationships")
        
        st.subheader("📦 Feature Distributions by Outcome")
        display_image('analysis/results/eda/feature_distribution_by_outcome.png', "How features differ between wins/losses")
    else:
        st.info("Run `python analysis/test.py` first to generate analysis results!")

elif analysis_section == "🏆 Key Findings":
    st.header("🏆 Key Research Findings")
    
    # Model Performance Results
    st.subheader("🤖 Model Performance Comparison")
    
    performance_data = {
        'Model': ['🏆 Random Forest', 'Neural Network', 'SVM', 'Decision Tree', 'KNN'],
        'Accuracy': ['87.5% ⭐', '84.5%', '83.5%', '79.5%', '79.0%'],
        'Status': ['✅ BEST - NOW USING!', '⚪ Good', '⚪ Previously used', '⚪ Basic', '⚪ Basic']
    }
    
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True)
    
    st.success("🎉 **Random Forest achieved highest accuracy and is now being used for predictions!**")
    
    # Top Features
    st.subheader("🎯 Most Important Prediction Factors")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Top 5 factors that determine fight outcomes:**
        
        1. **🏅 Win History Difference** (Most Important)
           - *Fighter with more wins has huge advantage*
        
        2. **👊 Striking Output** (Strikes/minute difference)
           - *More active strikers tend to win*
        
        3. **🎯 Striking Accuracy** 
           - *Precision beats volume*
        
        4. **⚖️ Weight Advantage**
           - *Bigger fighter has measurable edge*
        
        5. **🛡️ Defensive Skills**
           - *Good defense prevents losses*
        """)
    
    with col2:
        st.info("""
        **Model Upgrade:**
        
        ✅ **Before:** SVM (83.5%)
        🚀 **Now:** Random Forest (87.5%)
        
        📈 **Improvement:** +4% accuracy!
        """)

elif analysis_section == "📈 Performance Comparison":
    st.header("📈 Model Performance Comparison")
    
    if os.path.exists('analysis/results/comparisons'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy Comparison")
            display_image('analysis/results/comparisons/accuracy_comparison.png')
        
        with col2:
            st.subheader("F1 Score Comparison")
            display_image('analysis/results/comparisons/f1_comparison.png')
        
        st.subheader("Training Time Comparison")
        display_image('analysis/results/comparisons/time_comparison.png')
        
        st.subheader("Learning Curves")
        display_image('analysis/results/learning_curve/learning_curve.png')
    else:
        st.info("Run analysis first to see performance comparisons!")

elif analysis_section == "🔍 Feature Analysis":
    st.header("🔍 Feature Importance Analysis")
    
    if os.path.exists('analysis/results/feature_analysis'):
        st.markdown("Features ranked by importance in the Random Forest model:")
        
        if os.path.exists('analysis/results/feature_analysis/feature_importances.png'):
            display_image('analysis/results/feature_analysis/feature_importances.png')
        elif os.path.exists('analysis/results/feature_analysis/feature_coefficients.png'):
            display_image('analysis/results/feature_analysis/feature_coefficients.png')
    else:
        st.info("Feature analysis not available yet. Run the analysis script first!")

# Footer
st.markdown("---")
st.markdown("""
<div class="info-box">
    <p><strong>🚀 Upgraded Model:</strong> Now using Random Forest (87.5% accuracy) instead of SVM (83.5%) for better predictions!</p>
    <p><strong>📊 Unified Interface:</strong> Complete analysis and predictions in one place.</p>
    <p style="text-align:right;margin-top:20px;">© 2024 UFC Fight Predictor | Model: Random Forest 87.5% ⭐</p>
</div>
""", unsafe_allow_html=True)