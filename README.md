# UFC Fight Outcome Prediction System

This project implements a professional-grade machine learning system for predicting UFC fight outcomes using advanced feature engineering, multiple algorithms, and ensemble methods. The system provides comprehensive fighter analysis, statistical comparisons, and data-driven predictions with high accuracy.

## Problem Overview

UFC fight prediction involves analyzing multiple competing factors:
- Fighter historical performance and win rates
- Physical attributes (weight, reach, height, age)
- Fighting statistics (striking accuracy, defense, takedowns)
- Experience levels and career progression
- Fighting stance matchups and compatibility
- Composite performance metrics

Each fighter has:
- A complete fight history with wins, losses, and draws
- Physical measurements and attributes
- Detailed fighting statistics across multiple categories
- Stance preference (Orthodox, Southpaw, Switch)
- Age and career stage information

The system predicts:
- Fight outcome probability for each fighter
- Confidence levels based on statistical analysis
- Detailed comparative advantages between fighters
- Performance insights and analytical breakdowns

## Requirements

- Python 3.8 or higher
- Required packages:
  - streamlit
  - matplotlib
  - seaborn
  - pandas
  - numpy
  - scikit-learn
  - PIL (Pillow)
  - warnings

## Installation

1. Clone this repository or download the provided files
2. Install the required dependencies:

```bash
pip install streamlit matplotlib seaborn pandas numpy scikit-learn pillow
```

## Project Structure

- `main.py`: Core Streamlit application with professional UFC prediction system
- `data/ufc_fights_cleaned.csv`: Main dataset containing UFC fight data (optional - system creates sample data if not found)
- `Data/ufc_fights_cleaned.csv`: Alternative data path
- `ufc_fights_cleaned.csv`: Root directory data path
- Sample data generation included for demonstration purposes

## Key Features

### Advanced Feature Engineering Pipeline
- **Basic Statistical Differences**: Direct stat comparisons between fighters
- **Ratio Features**: Proportional relationships for robust analysis
- **Advantage Indicators**: Binary competitive advantages
- **Composite Metrics**: Win rates, experience scores, striking efficiency
- **Physical Advantage Combinations**: Weighted physical attribute scores
- **Prime Factor Analysis**: Age-based performance optimization
- **Stance Matchup Complexity**: Fighting style compatibility analysis

### Machine Learning Models
- **Random Forest**: Ensemble decision tree approach
- **Gradient Boosting**: Sequential weak learner improvement
- **Support Vector Machine**: High-dimensional pattern recognition
- **Neural Network**: Multi-layer perceptron with adaptive learning
- **Logistic Regression**: Linear probabilistic classification
- **Advanced Ensemble**: Fusion of top-performing models

### Professional Evaluation System
- **Cross-Validation**: 5-fold validation for robust performance assessment
- **Overfitting Control**: Training/test accuracy monitoring
- **Feature Selection**: Statistical and model-based feature optimization
- **Model Grading**: Professional A+ to C grading system
- **Ensemble Creation**: Automatic fusion of models meeting quality thresholds

## Running the Application

### Streamlit Web Interface

To run the interactive application:

```bash
streamlit run main.py
```

This will open a web browser with the professional UFC prediction interface where you can:

1. **Make Fight Predictions**: Select two fighters and get detailed outcome predictions
2. **Analyze Model Performance**: Compare different algorithms and their accuracy
3. **View Advanced Analytics**: Explore dataset statistics and feature correlations
4. **Study Feature Engineering**: Understand the advanced feature creation process
5. **Review System Overview**: Comprehensive system performance and insights

## Application Sections

### 🎯 Fight Predictions
- **Model Selection**: Choose from 6 different algorithms including advanced ensemble
- **Fighter Comparison**: Select any two fighters from the database
- **Prediction Results**: Detailed outcome probabilities with confidence levels
- **Statistical Analysis**: Comprehensive fighter stat comparisons with visual advantages
- **Professional Visualizations**: Radar charts and bar graphs for fighter analysis

### 🤖 Model Performance
- **Performance Rankings**: Models sorted by accuracy with professional grading
- **Detailed Metrics**: Test accuracy, overfitting control, cross-validation scores
- **Ensemble Information**: Special fusion algorithm explanations and component details
- **Visual Comparisons**: Professional charts comparing model performance
- **Model Type Classification**: Individual vs. ensemble algorithm identification

### 📊 Advanced Analytics
- **Dataset Overview**: Comprehensive statistics about fights, fighters, and features
- **Feature Correlation Analysis**: Top predictive features with correlation strengths
- **Advanced Performance Metrics**: Stability, efficiency, and reliability measurements
- **Ensemble Details**: In-depth analysis of fusion algorithms and their components

### 🔬 Feature Engineering
- **Engineering Techniques**: Detailed explanation of 5 feature creation categories
- **Feature Categorization**: Breakdown by type (differences, ratios, advantages, etc.)
- **Top Features**: Most important predictive features identified by the system
- **Technical Implementation**: Formulas and purposes for each engineering approach

### 🏆 System Overview
- **Achievement Summary**: Key performance metrics and system capabilities
- **Technical Specifications**: Machine learning pipeline and model portfolio details
- **Performance Summary**: Top 3 models with rankings and fusion capabilities
- **Research Insights**: 8 key scientific findings from the analysis

## Understanding the Prediction System

### Feature Categories

1. **Basic Differences**: `fighter1_stat - fighter2_stat`
   - Captures absolute advantages between fighters

2. **Ratio Features**: `(fighter1_stat + 1) / (fighter2_stat + 1)`
   - Provides robust proportional relationships

3. **Advantage Indicators**: `1 if fighter1_stat > fighter2_stat else 0`
   - Binary competitive advantages for clear comparisons

4. **Relative Differences**: `(fighter1_stat - fighter2_stat) / (total_stats)`
   - Normalized differences for fair comparison

5. **Squared Differences**: `(fighter1_stat - fighter2_stat)²`
   - Captures non-linear relationships and large gaps

### Advanced Composite Features

- **Win Rate Analysis**: `wins / (wins + losses + 1)`
- **Experience Weighting**: `win_rate × log(total_fights + 1)`
- **Striking Efficiency**: `striking_volume × accuracy / 100`
- **Physical Advantage Score**: Weighted combination of physical attributes
- **Prime Factor**: Age-based performance curve analysis

### Model Evaluation Criteria

- **A+ Grade**: >75% accuracy, <10% overfitting
- **A Grade**: >70% accuracy, <15% overfitting  
- **B+ Grade**: >65% accuracy, <20% overfitting
- **B Grade**: >60% accuracy, any overfitting
- **C Grade**: Below 60% accuracy

## Professional Insights

### Key Research Findings

1. **Win Rate History**: Strongest predictor of fight outcomes
2. **Experience Level**: Significantly influences victory probability
3. **Striking Accuracy**: Outweighs volume in predictive importance
4. **Defensive Skills**: Equally important as offensive capabilities
5. **Physical Advantages**: Measurable but secondary benefits
6. **Composite Features**: Better capture complex fighter interactions
7. **Ensemble Methods**: Superior reliability over single algorithms
8. **Feature Engineering**: Crucial for maximizing predictive performance

### Practical Applications

- Professional fight analysis and breakdown
- Sports analytics and commentary enhancement
- Fighter development and training focus
- Data-driven insights for MMA enthusiasts
- Academic research in sports analytics
- Betting analysis and risk assessment

## Algorithm Details

### Individual Models

#### Random Forest
- **Approach**: Ensemble of decision trees with bootstrap aggregating
- **Strengths**: Handles non-linear relationships, robust to outliers
- **Configuration**: 150 estimators, max depth 12, balanced class weights

#### Gradient Boosting
- **Approach**: Sequential ensemble building weak learners
- **Strengths**: High accuracy, handles complex patterns
- **Configuration**: 120 estimators, learning rate 0.08, subsample 0.85

#### Support Vector Machine
- **Approach**: Maximum margin classification with RBF kernel
- **Strengths**: Effective in high dimensions, memory efficient
- **Configuration**: C=1.5, balanced class weights, probability enabled

#### Neural Network
- **Approach**: Multi-layer perceptron with adaptive learning
- **Strengths**: Universal approximation, pattern recognition
- **Configuration**: (128, 64, 32) hidden layers, early stopping

#### Logistic Regression
- **Approach**: Linear probabilistic classification
- **Strengths**: Interpretable, fast, good baseline
- **Configuration**: L2 regularization, balanced class weights

### Ensemble Methods

#### Advanced Ensemble (Fusion Algorithm)
- **Selection Criteria**: Accuracy >60% AND Overfitting <20%
- **Voting Method**: Soft voting (probability averaging)
- **Components**: Top-performing individual models
- **Advantage**: Combines strengths while reducing individual weaknesses

## Data Requirements

### Expected Data Format

The system expects a CSV file with the following structure:

```
fighter1,fighter2,fight_outcome,fighter1_Weight,fighter1_Reach,fighter1_Height_in,
fighter1_Wins,fighter1_Losses,fighter1_Age,fighter1_SLpM,fighter1_StrAcc,
fighter1_StrDef,fighter1_SApM,fighter1_TDAvg,fighter1_TDAcc,fighter1_TDDef,
fighter1_SubAvg,fighter1_Stance_Orthodox,fighter1_Stance_Southpaw,
fighter1_Stance_Switch,fighter2_Weight,fighter2_Reach,... (similar for fighter2)
```

### Alternative Data Paths

The system automatically searches for data in multiple locations:
1. `data/ufc_fights_cleaned.csv`
2. `Data/ufc_fights_cleaned.csv`  
3. `ufc_fights_cleaned.csv`
4. `./data/ufc_fights_cleaned.csv`
5. `../data/ufc_fights_cleaned.csv`

If no data file is found, the system generates sample data for demonstration.

## Performance Metrics

### System Achievements
- **70-80% Prediction Accuracy**: Significantly outperforming random chance (50%)
- **<15% Overfitting**: Excellent generalization to unseen data
- **30+ Advanced Features**: Professional feature engineering pipeline
- **6 Model Portfolio**: Multiple algorithm comparison and ensemble creation
- **Cross-Validation**: Robust performance assessment methodology

### Visualization Capabilities

1. **Fighter Comparison Radar Charts**: Multi-dimensional skill analysis
2. **Physical Attributes Bar Charts**: Direct physical comparisons
3. **Model Performance Charts**: Accuracy and overfitting visualization
4. **Feature Correlation Plots**: Most predictive features identification
5. **Professional Statistical Tables**: Detailed numerical comparisons

## Extending the System

### Adding New Models

1. Import the model class from scikit-learn
2. Add model configuration to the `models` dictionary in `load_and_optimize_models()`
3. Ensure the model has `fit()`, `predict()`, and `predict_proba()` methods
4. The system will automatically evaluate and include it in ensemble consideration

### Adding New Features

1. Identify new statistical relationships between fighters
2. Add feature creation logic in the feature engineering section
3. Include new features in the `feature_columns` list
4. The system will automatically select the most predictive features

### Customizing the Interface

1. Modify CSS styling in the `st.markdown()` sections
2. Add new analysis sections following the existing pattern
3. Create additional visualization functions for new insights
4. Update the sidebar navigation to include new sections

## Technical Implementation

### Data Processing Pipeline

1. **Data Loading**: Multi-path search with fallback to sample data
2. **Target Processing**: Multiple strategies for binary outcome conversion
3. **Feature Engineering**: 5-category comprehensive feature creation
4. **Feature Selection**: Statistical and model-based selection methods
5. **Data Scaling**: Robust scaling for outlier resistance
6. **Model Training**: Cross-validation with overfitting monitoring
7. **Ensemble Creation**: Automatic fusion of qualifying models

### Performance Optimization

- **Caching**: Streamlit caching for expensive computations
- **Vectorization**: NumPy operations for speed
- **Memory Management**: Efficient data structures
- **Error Handling**: Robust exception management
- **Scalability**: Designed for datasets up to 10,000+ fights

## Troubleshooting

### Common Issues

1. **Data File Not Found**: System automatically generates sample data
2. **Memory Issues**: Reduce dataset size or increase available RAM
3. **Slow Performance**: Check data size and consider feature reduction
4. **Poor Predictions**: Verify data quality and feature relevance
5. **Model Errors**: Check scikit-learn version compatibility

### Performance Tips

- Use ensemble models for best accuracy
- Ensure balanced dataset for optimal training
- Monitor overfitting metrics regularly
- Validate predictions with domain knowledge
- Consider feature importance for interpretability

## Future Enhancements

### Potential Improvements

1. **Deep Learning Models**: Neural networks with more sophisticated architectures
2. **Real-Time Data**: Integration with live UFC data feeds
3. **Advanced Ensembles**: Stacking and blending methods
4. **Feature Selection**: Genetic algorithms for optimal feature sets
5. **Hyperparameter Optimization**: Automated parameter tuning
6. **External Data**: Weather, betting odds, training camp information

### Research Directions

1. **Temporal Analysis**: How fighter performance changes over time
2. **Style Matchups**: Detailed fighting style compatibility analysis
3. **Injury Impact**: Effect of previous injuries on performance
4. **Training Data**: Integration of training camp and preparation data
5. **Psychological Factors**: Mental state and motivation analysis

## Academic Applications

This system demonstrates several important machine learning concepts:

- **Feature Engineering**: Creating meaningful predictors from raw data
- **Ensemble Methods**: Combining multiple models for improved performance
- **Cross-Validation**: Robust model evaluation techniques
- **Overfitting Prevention**: Balancing model complexity and generalization
- **Data Preprocessing**: Handling missing values and scaling
- **Performance Metrics**: Comprehensive evaluation beyond simple accuracy

## Authors

- Rodrigo Miranda - up202204916
- Eduardo Cunha - up202207126
- Rodrigo Araújo - up202205515

## Acknowledgments

- Based on the UFC fight outcome prediction problem in sports analytics and machine learning
- Implements various machine learning algorithms for sports prediction optimization