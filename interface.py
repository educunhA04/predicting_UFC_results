import streamlit as st
import matplotlib.pyplot as plt
from prediction import load_data, train_model, compare_fighters, create_comparison_charts

# Set page config
st.set_page_config(
    page_title="UFC Fight Outcome Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .header-style {
        font-size: 24px;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #0E1117;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .fighter-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .stat-diff-positive {
        color: #00FF00;
        font-weight: bold;
    }
    .stat-diff-negative {
        color: #FF0000;
        font-weight: bold;
    }
    .stat-diff-neutral {
        color: #CCCCCC;
    }
    .prediction-banner {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8C8C 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
df, all_fighters = load_data()
model, scaler, feature_columns = train_model(df)

# Streamlit UI
st.title("🥊 UFC Fight Outcome Predictor")
st.markdown("""
<div style="background-color:#262730;padding:20px;border-radius:10px;">
    <p style="font-size:16px;">This app predicts the outcome of a UFC fight between two selected fighters using a 
    Support Vector Machine (SVM) model with <strong>83.5% accuracy</strong>, based on fighter statistics and historical fight data.</p>
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

if st.button("Predict Fight Outcome", type="primary"):
    if fighter1 == fighter2:
        st.error("Please select two different fighters.")
    else:
        try:
            result = compare_fighters(df, model, scaler, feature_columns, fighter1, fighter2)
            
            st.markdown("---")
            st.markdown(f'<div class="header-style">🏆 Prediction Results</div>', unsafe_allow_html=True)
            
            # Display prediction in cards
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{fighter1}</h3>', unsafe_allow_html=True)
                st.metric("Win Probability", f"{result['fighter1_prob']*100:.1f}%", delta=f"+{(result['fighter1_prob']-0.5)*100:.1f}% vs average")
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>{fighter2}</h3>', unsafe_allow_html=True)
                st.metric("Win Probability", f"{result['fighter2_prob']*100:.1f}%", delta=f"+{(result['fighter2_prob']-0.5)*100:.1f}% vs average")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display predicted winner banner
            predicted_winner = fighter1 if result['prediction'] == 1 else fighter2
            confidence = max(result['fighter1_prob'], result['fighter2_prob'])
            st.markdown(
                f'<div class="prediction-banner">'
                f'<h2 style="color:white;text-align:center;">PREDICTED WINNER: {predicted_winner.upper()} ({(confidence*100):.1f}% confidence)</h2>'
                f'</div>', 
                unsafe_allow_html=True
            )
            
            # Visualizations
            st.markdown("---")
            st.markdown(f'<div class="header-style">📊 Fighter Comparison Visualizations</div>', unsafe_allow_html=True)
            
            radar_fig, bar_fig = create_comparison_charts(fighter1, fighter2, result['fighter1_stats'], result['fighter2_stats'])
            st.pyplot(radar_fig)
            st.pyplot(bar_fig)
            
            # Stats comparison
            st.markdown("---")
            st.markdown(f'<div class="header-style">📝 Detailed Statistics Comparison</div>', unsafe_allow_html=True)
            
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
            
            # Display each stat with colored difference
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
                            if diff > 0:
                                st.markdown(f"<span class='stat-diff-positive'>+{diff:.2f}</span>", unsafe_allow_html=True)
                            elif diff < 0:
                                st.markdown(f"<span class='stat-diff-negative'>{diff:.2f}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span class='stat-diff-neutral'>{diff:.2f}</span>", unsafe_allow_html=True)
            
            # Additional info in cards
            st.markdown("---")
            st.markdown(f'<div class="header-style">ℹ️ Additional Fighter Information</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="fighter-card">'
                            f'<h3>{fighter1}</h3>'
                            f'<p><strong>Stance:</strong> {result["fighter1_stance"]}</p>'
                            f'<p><strong>Age:</strong> {result["fighter1_age"]:.1f} years</p>'
                            f'<p><strong>Record:</strong> {result["fighter1_stats"]["Wins"]:.0f}-{result["fighter1_stats"]["Losses"]:.0f}-{result["fighter1_stats"]["Draws"]:.0f}</p>'
                            f'</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="fighter-card">'
                            f'<h3>{fighter2}</h3>'
                            f'<p><strong>Stance:</strong> {result["fighter2_stance"]}</p>'
                            f'<p><strong>Age:</strong> {result["fighter2_age"]:.1f} years</p>'
                            f'<p><strong>Record:</strong> {result["fighter2_stats"]["Wins"]:.0f}-{result["fighter2_stats"]["Losses"]:.0f}-{result["fighter2_stats"]["Draws"]:.0f}</p>'
                            f'</div>', unsafe_allow_html=True)
            
            # Key metrics explanation in expander
            with st.expander("📖 Key Metrics Explanation"):
                st.markdown("""
                - **SLpM:** Significant Strikes Landed per Minute
                - **StrAcc:** Striking Accuracy (% of strikes that land)
                - **StrDef:** Striking Defense (% of opponent's strikes blocked/avoided)
                - **TDAvg:** Average Takedowns per 15 minutes
                - **TDAcc:** Takedown Accuracy (% of takedown attempts that succeed)
                - **TDDef:** Takedown Defense (% of opponent's takedowns defended)
                - **SubAvg:** Average Submissions Attempted per 15 minutes
                """)
            
        except Exception as e:
            st.error(f"Error processing fighters: {str(e)}")
            st.info("Please try different fighters. Some fighter data may be incomplete.")

# Footer
st.markdown("---")
st.markdown("""
<div style="background-color:#262730;padding:20px;border-radius:10px;font-size:14px;">
    <p><strong>Note:</strong> This prediction is based on historical data and statistical modeling (SVM with 83.5% accuracy). 
    Actual fight outcomes may vary due to factors not captured in the data like injuries, recent form, or stylistic matchups.</p>
    <p style="text-align:right;margin-top:20px;">© 2023 UFC Fight Predictor | Model Accuracy: 83.5%</p>
</div>
""", unsafe_allow_html=True)