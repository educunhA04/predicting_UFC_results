# UFC Fight Outcome Prediction System

> Professional-grade machine learning system for predicting UFC fight outcomes using advanced feature engineering and ensemble methods

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Framework-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Excellence](https://img.shields.io/badge/FEUP%20AI%20Course-20%2F20-green.svg)](https://sigarra.up.pt/feup/)

A comprehensive machine learning system that analyzes UFC fight data to predict outcomes with 70-80% accuracy using advanced feature engineering, multiple algorithms, and ensemble methods. The system provides professional fighter analysis, statistical comparisons, and data-driven predictions with high confidence levels.

## Key Features

- **Six Advanced Machine Learning Models**: Random Forest, Gradient Boosting, SVM, Neural Network, Logistic Regression, and Advanced Ensemble
- **Professional Feature Engineering**: 30+ engineered features across 5 categories for maximum predictive power
- **Interactive Web Interface**: Real-time predictions with comprehensive fighter analysis and visualizations
- **Cross-Validation System**: Robust 5-fold validation with overfitting control and professional model grading
- **Ensemble Intelligence**: Automatic fusion of top-performing models meeting quality thresholds
- **Advanced Analytics Dashboard**: Dataset insights, feature correlations, and model performance comparisons

## Problem Overview

UFC fight prediction involves analyzing multiple competing factors to determine fight outcomes:

| Analysis Category | Key Components |
|------------------|----------------|
| **Fighter Performance** | Win rates, fight history, experience levels, career progression |
| **Physical Attributes** | Weight, reach, height, age, physical advantages |
| **Fighting Statistics** | Striking accuracy/defense, takedown rates, submission averages |
| **Style Analysis** | Stance matchups, fighting style compatibility |
| **Composite Metrics** | Engineered features combining multiple statistical elements |

### System Capabilities

The system predicts:
- **Fight Outcome Probabilities**: Detailed percentage chances for each fighter
- **Confidence Levels**: Statistical reliability of predictions based on data quality
- **Comparative Analysis**: Detailed breakdown of advantages between fighters
- **Performance Insights**: Professional analytical breakdowns and fighter assessments

## Quick Start

### Prerequisites

```bash
Python 3.8+
pip install streamlit matplotlib seaborn pandas numpy scikit-learn pillow
```

### Installation

```bash
git clone https://github.com/yourusername/ufc-fight-prediction.git
cd ufc-fight-prediction
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run main.py
```

## Project Structure

```
ufc-fight-prediction/
├── main.py                     # Core Streamlit application
├── data/                       # Data directory
│   └── ufc_fights_cleaned.csv  # Main UFC dataset (optional)
├── Data/                       # Alternative data path
│   └── ufc_fights_cleaned.csv
├── ufc_fights_cleaned.csv      # Root directory data
└── requirements.txt            # Dependencies
```

## Machine Learning Pipeline

### Advanced Feature Engineering

The system creates 30+ engineered features across five categories:

#### 1. Basic Statistical Differences
```python
fighter1_stat - fighter2_stat
```
Direct stat comparisons capturing absolute advantages

#### 2. Ratio Features  
```python
(fighter1_stat + 1) / (fighter2_stat + 1)
```
Proportional relationships for robust analysis

#### 3. Advantage Indicators
```python
1 if fighter1_stat > fighter2_stat else 0
```
Binary competitive advantages for clear comparisons

#### 4. Relative Differences
```python
(fighter1_stat - fighter2_stat) / (total_stats)
```
Normalized differences for fair comparison

#### 5. Squared Differences
```python
(fighter1_stat - fighter2_stat)²
```
Non-linear relationships and performance gaps

### Advanced Composite Features

- **Win Rate Analysis**: `wins / (wins + losses + 1)`
- **Experience Weighting**: `win_rate × log(total_fights + 1)`  
- **Striking Efficiency**: `striking_volume × accuracy / 100`
- **Physical Advantage Score**: Weighted combination of physical attributes
- **Prime Factor**: Age-based performance curve analysis
- **Stance Matchup Complexity**: Fighting style compatibility metrics

## Model Portfolio

### Individual Models

| Algorithm | Approach | Key Strengths | Configuration |
|-----------|----------|---------------|---------------|
| **Random Forest** | Ensemble decision trees | Non-linear patterns, outlier robust | 150 estimators, max depth 12 |
| **Gradient Boosting** | Sequential weak learners | High accuracy, complex patterns | 120 estimators, learning rate 0.08 |
| **Support Vector Machine** | Maximum margin classification | High-dimensional effectiveness | RBF kernel, C=1.5 |
| **Neural Network** | Multi-layer perceptron | Universal approximation | (128,64,32) layers, early stopping |
| **Logistic Regression** | Linear probabilistic | Interpretable, fast baseline | L2 regularization, balanced weights |

### Advanced Ensemble System

**Selection Criteria**: Accuracy >60% AND Overfitting <20%
- **Voting Method**: Soft voting (probability averaging)
- **Quality Control**: Professional grading system (A+ to C grades)
- **Automatic Fusion**: Combines strengths while reducing individual weaknesses

## Professional Evaluation System

### Model Grading Criteria

| Grade | Accuracy Threshold | Overfitting Limit | Quality Level |
|-------|-------------------|-------------------|---------------|
| **A+** | >75% | <10% | Exceptional |
| **A** | >70% | <15% | Excellent |
| **B+** | >65% | <20% | Very Good |
| **B** | >60% | Any | Good |
| **C** | <60% | Any | Acceptable |

### Performance Metrics

- **Cross-Validation**: 5-fold validation for robust assessment
- **Overfitting Control**: Training/test accuracy monitoring  
- **Feature Selection**: Statistical and model-based optimization
- **Ensemble Creation**: Automatic fusion meeting quality thresholds

## Application Interface

### Fight Predictions
- **Model Selection**: Choose from 6 algorithms including advanced ensemble
- **Fighter Database**: Complete UFC fighter roster with historical data
- **Prediction Results**: Detailed probabilities with confidence levels
- **Visual Analysis**: Radar charts and statistical comparisons
- **Professional Insights**: Comprehensive fighter breakdowns

### Model Performance Dashboard
- **Performance Rankings**: Models sorted by accuracy with professional grades
- **Detailed Metrics**: Cross-validation scores and overfitting analysis
- **Ensemble Intelligence**: Fusion algorithm details and component analysis
- **Visual Comparisons**: Professional performance visualization charts

### Advanced Analytics
- **Dataset Overview**: Comprehensive fight and fighter statistics
- **Feature Analysis**: Top predictive features with correlation strengths
- **Performance Metrics**: Stability, efficiency, and reliability measurements
- **Research Insights**: Key findings from comprehensive data analysis

### Technical Documentation
- **Feature Engineering**: Detailed explanation of 5 engineering categories
- **Implementation Details**: Formulas and technical specifications
- **System Architecture**: Machine learning pipeline and model integration
- **Research Findings**: 8 key insights from scientific analysis

## Key Research Findings

1. **Win Rate History**: Strongest single predictor of fight outcomes
2. **Experience Level**: Significantly influences victory probability  
3. **Striking Accuracy**: Outweighs volume in predictive importance
4. **Defensive Skills**: Equally important as offensive capabilities
5. **Physical Advantages**: Measurable but secondary to skill metrics
6. **Composite Features**: Superior to individual stats for prediction
7. **Ensemble Methods**: Consistently outperform individual algorithms
8. **Feature Engineering**: Critical for maximizing predictive performance

## Data Requirements

### Expected Data Format

```csv
fighter1,fighter2,fight_outcome,fighter1_Weight,fighter1_Reach,fighter1_Height_in,
fighter1_Wins,fighter1_Losses,fighter1_Age,fighter1_SLpM,fighter1_StrAcc,
fighter1_StrDef,fighter1_SApM,fighter1_TDAvg,fighter1_TDAcc,fighter1_TDDef,
fighter1_SubAvg,fighter1_Stance_Orthodox,fighter1_Stance_Southpaw,
fighter1_Stance_Switch,fighter2_Weight,fighter2_Reach,... (similar for fighter2)
```

### Automatic Data Discovery

The system searches multiple locations:
1. `data/ufc_fights_cleaned.csv`
2. `Data/ufc_fights_cleaned.csv`
3. `ufc_fights_cleaned.csv`
4. `./data/ufc_fights_cleaned.csv`
5. `../data/ufc_fights_cleaned.csv`

**Fallback**: Generates sample data for demonstration if no file found

## Performance Achievements

- **70-80% Prediction Accuracy**: Significantly outperforming random chance (50%)
- **<15% Average Overfitting**: Excellent generalization to unseen data
- **30+ Advanced Features**: Professional feature engineering pipeline
- **6-Model Portfolio**: Comprehensive algorithm comparison and ensemble creation
- **Cross-Validation Methodology**: Robust performance assessment system

## Practical Applications

| Domain | Use Case |
|--------|----------|
| **Professional Analysis** | Fight breakdowns and commentary enhancement |
| **Sports Analytics** | Data-driven insights for MMA analysis |
| **Fighter Development** | Training focus and performance optimization |
| **Academic Research** | Sports analytics and machine learning studies |
| **Risk Assessment** | Statistical analysis for informed decision-making |

## Extending the System

### Adding New Models

```python
# Add to models dictionary in load_and_optimize_models()
models = {
    'your_model': YourModel(
        # configuration parameters
    )
}
```

### Adding New Features

1. Identify statistical relationships between fighters
2. Add feature creation logic in engineering section
3. Include in `feature_columns` list
4. System automatically selects most predictive features

### Interface Customization

1. Modify CSS styling in `st.markdown()` sections
2. Add analysis sections following existing patterns
3. Create visualization functions for new insights
4. Update sidebar navigation for new sections

## Technical Implementation

### Data Processing Pipeline

1. **Multi-Path Data Loading** with sample data fallback
2. **Target Processing** with multiple binary outcome strategies  
3. **5-Category Feature Engineering** with comprehensive feature creation
4. **Feature Selection** using statistical and model-based methods
5. **Robust Data Scaling** for outlier resistance
6. **Cross-Validated Training** with overfitting monitoring
7. **Automatic Ensemble Creation** for qualifying models

### Performance Optimization

- **Streamlit Caching**: Efficient computation management
- **Vectorized Operations**: NumPy optimization for speed
- **Memory Management**: Efficient data structures and processing
- **Error Handling**: Robust exception management system
- **Scalability**: Designed for 10,000+ fight datasets

## Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Data File Not Found** | System generates sample data automatically |
| **Memory Constraints** | Reduce dataset size or increase available RAM |
| **Slow Performance** | Check data size, consider feature reduction |
| **Poor Predictions** | Verify data quality and feature relevance |
| **Model Errors** | Check scikit-learn version compatibility |

## Future Enhancements

### Technical Improvements
- **Deep Learning Architecture**: Advanced neural network models
- **Real-Time Data Integration**: Live UFC data feeds
- **Advanced Ensemble Methods**: Stacking and blending techniques
- **Automated Hyperparameter Optimization**: Grid search and Bayesian optimization
- **External Data Integration**: Training camp, injury, and betting data

### Research Directions  
- **Temporal Analysis**: Fighter performance evolution over time
- **Advanced Style Analysis**: Detailed fighting style compatibility
- **Injury Impact Assessment**: Previous injury effects on performance
- **Psychological Factor Integration**: Mental state and motivation analysis

## Contributors

- **[Rodrigo Miranda](https://github.com/h0leee)** - up202204916
- **[Eduardo Cunha](https://github.com/educunhA04)** - up202207126
- **[Rodrigo Araújo](https://github.com/rodrigoaraujo9)** - up202205515

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Rodrigo Miranda, Eduardo Cunha, Rodrigo Araújo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Academic Excellence

This project achieved a **perfect score of 20/20** in the Artificial Intelligence course at [FEUP](https://sigarra.up.pt/feup/) (Faculdade de Engenharia da Universidade do Porto), demonstrating excellence in:
- Advanced machine learning implementation and optimization
- Professional feature engineering and data processing
- Ensemble method development and evaluation
- Comprehensive system design and user experience
- Scientific analysis and research methodology

**Two consecutive 20/20 projects** showcase consistent excellence in AI coursework and advanced machine learning applications.

## Acknowledgments

- Inspired by real-world sports analytics and machine learning challenges
- Implements state-of-the-art machine learning techniques for sports prediction
- Built upon established principles in ensemble methods and feature engineering
- Developed as part of advanced AI coursework at FEUP

---

**If this project helped you, please give it a star!**

[Report Bug](../../issues) • [Request Feature](../../issues) • [Documentation](../../wiki)
