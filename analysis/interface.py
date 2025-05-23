import streamlit as st
import os
from PIL import Image
import pandas as pd

# Set page config
st.set_page_config(
    page_title="UFC Fight Outcome Prediction Analysis",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🥊 UFC Fight Outcome Prediction Analysis")
st.markdown("---")
st.markdown("""
This dashboard displays the results from the UFC fight outcome prediction machine learning analysis.
The analysis compares different supervised learning models to predict fight outcomes based on fighter statistics.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
analysis_section = st.sidebar.selectbox(
    "Select Analysis Section:",
    ["Overview", "Exploratory Data Analysis", "Model Performance", "Model Comparisons", "Learning Curves", "Feature Analysis"]
)

# Function to display image if it exists
def display_image(image_path, caption="", width=None):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=caption, width=width)
    else:
        st.error(f"Image not found: {image_path}")
        st.info("Make sure to run the main analysis script (test.py) first to generate the plots.")

# Function to check if results directory exists
def check_results_directory():
    if not os.path.exists('results'):
        st.error("Results directory not found!")
        st.info("Please run the main analysis script (test.py) first to generate the results.")
        return False
    return True

# Main content based on selected section
if analysis_section == "Overview":
    st.header("📊 Analysis Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        st.markdown("""
        - **Dataset Size**: 2000 simulated UFC fights
        - **Features**: Fighter statistics differences (weight, height, reach, wins, losses, striking stats, etc.)
        - **Target**: Binary classification (Fighter 1 wins vs Fighter 2 wins)
        - **Models Tested**: Decision Tree, Random Forest, SVM, Neural Network, K-Nearest Neighbors
        """)
    
    with col2:
        st.subheader("Key Features")
        st.markdown("""
        - **Physical Stats**: Weight, Height, Reach differences
        - **Fight Record**: Wins, Losses, Draws differences  
        - **Striking Stats**: Accuracy, Defense, Strikes per minute
        - **Grappling Stats**: Takedown accuracy, Submission average
        - **Fighting Stance**: Orthodox, Southpaw, Switch differences
        """)
    
    if check_results_directory():
        st.subheader("Class Distribution")
        display_image('results/eda/class_distribution.png', "Distribution of fight outcomes in the dataset")

elif analysis_section == "Exploratory Data Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    if not check_results_directory():
        st.stop()
    
    st.subheader("Feature Correlations")
    st.markdown("This heatmap shows how different fighter statistic differences correlate with fight outcomes and each other.")
    display_image('results/eda/correlation_matrix.png', "Correlation matrix of features with fight outcomes")
    
    st.subheader("Feature Distributions by Outcome")
    st.markdown("These box plots show how feature values differ between wins and losses for Fighter 1.")
    display_image('results/eda/feature_distribution_by_outcome.png', "Distribution of top features by fight outcome")

elif analysis_section == "Model Performance":
    st.header("🤖 Individual Model Performance")
    
    if not check_results_directory():
        st.stop()
    
    # Model selection
    models = ['Decision_Tree', 'Random_Forest', 'Support_Vector_Machine', 'Neural_Network', 'K-Nearest_Neighbors']
    model_names = ['Decision Tree', 'Random Forest', 'Support Vector Machine', 'Neural Network', 'K-Nearest Neighbors']
    
    selected_model = st.selectbox("Select a model to view detailed results:", model_names)
    model_file_name = models[model_names.index(selected_model)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Confusion Matrix - {selected_model}")
        display_image(f'results/models/confusion_matrix_{model_file_name}.png', 
                     f"Confusion matrix showing prediction accuracy for {selected_model}")
    
    with col2:
        st.subheader(f"ROC Curve - {selected_model}")
        roc_path = f'results/models/roc_curve_{model_file_name}.png'
        if os.path.exists(roc_path):
            display_image(roc_path, f"ROC curve showing classifier performance for {selected_model}")
        else:
            st.info("ROC curve not available for this model (may not support probability predictions)")

elif analysis_section == "Model Comparisons":
    st.header("📈 Model Comparison Results")
    
    if not check_results_directory():
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy Comparison")
        display_image('results/comparisons/accuracy_comparison.png', "Accuracy scores across all models")
    
    with col2:
        st.subheader("F1 Score Comparison")
        display_image('results/comparisons/f1_comparison.png', "F1 scores across all models")
    
    st.subheader("Training and Testing Time Comparison")
    display_image('results/comparisons/time_comparison.png', "Computational efficiency comparison across models")
    
    st.markdown("""
    **Key Insights:**
    - Compare model accuracy and F1 scores to identify the best performer
    - Consider the trade-off between performance and computational time
    - F1 score is particularly important for balanced evaluation of precision and recall
    """)

elif analysis_section == "Learning Curves":
    st.header("📚 Learning Curve Analysis")
    
    if not check_results_directory():
        st.stop()
    
    st.markdown("""
    The learning curve shows how the best performing model's accuracy changes with the amount of training data.
    This helps identify if the model would benefit from more data or if it's overfitting.
    """)
    
    display_image('results/learning_curve/learning_curve.png', "Learning curves for the best performing model")
    
    st.markdown("""
    **How to interpret:**
    - **Gap between training and validation scores**: Large gap indicates overfitting
    - **Converging scores**: Good sign that the model generalizes well
    - **Plateau**: Model has reached its learning capacity with current features
    """)

elif analysis_section == "Feature Analysis":
    st.header("🎯 Feature Importance Analysis")
    
    if not check_results_directory():
        st.stop()
    
    st.markdown("""
    This section shows which fighter statistic differences are most important for predicting fight outcomes.
    The analysis is based on the best performing model.
    """)
    
    # Check for both possible file names
    feature_importance_path = 'results/feature_analysis/feature_importances.png'
    feature_coefficients_path = 'results/feature_analysis/feature_coefficients.png'
    
    if os.path.exists(feature_importance_path):
        st.subheader("Feature Importances")
        display_image(feature_importance_path, "Most important features for predicting fight outcomes")
    elif os.path.exists(feature_coefficients_path):
        st.subheader("Feature Coefficients")
        display_image(feature_coefficients_path, "Feature coefficients from the best model")
    else:
        st.info("Feature analysis plot not found. This may be generated depending on the best performing model type.")
    
    st.markdown("""
    **Expected Important Features (based on domain knowledge):**
    - **Wins Difference**: More experienced fighters tend to win
    - **Weight/Physical Advantages**: Size and reach advantages
    - **Striking Statistics**: Accuracy and defensive capabilities
    - **Fighting Style**: Stance matchups can influence outcomes
    """)

# Footer
st.markdown("---")
st.markdown("""
**About this Analysis:**
This dashboard presents results from a supervised learning analysis of UFC fight outcomes. 
The analysis uses simulated data based on real UFC fighter statistics and compares multiple 
machine learning algorithms to predict fight winners.

Run `test.py` to generate the analysis results, then use this Streamlit app to explore the findings.
""")

# Instructions for running
with st.expander("How to use this dashboard"):
    st.markdown("""
    1. **First time setup:**
       - Make sure you have all required packages installed: `pip install streamlit pandas pillow`
       - Run the main analysis script: `python test.py`
       - This will generate all the plots and results in the `results/` directory
    
    2. **Launch the dashboard:**
       - Run: `streamlit run streamlit_app.py`
       - The dashboard will open in your web browser
    
    3. **Navigate the results:**
       - Use the sidebar to switch between different analysis sections
       - Each section shows relevant plots and insights from the machine learning analysis
    """)