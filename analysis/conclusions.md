# 1. Dataset
## 1.1 Database Features


| Attribute              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| A - fighter1           | Name of the first fighter.                                                  |
| B - fighter2           | Name of the second fighter.                                                 |
| C - event              | Name of the UFC event where the fight took place.                           |
| D - fight_outcome      | Indicates which fighter won the fight.                                      |
| E - origin_fight_url   | URL linking to the original source of the fight details.                    |
| F - fighter1_Name      | Full name of fighter 1.                                                     |
| G - fighter1_Nickname  | Nickname of fighter 1.                                                      |
| H - fighter1_Record    | Fight record (wins-losses-draws) of fighter 1.                              |
| I - fighter1_Height    | Height of fighter 1.                                                        |
| J - fighter1_Weight    | Weight of fighter 1.                                                        |
| K - fighter1_Reach     | Reach of fighter 1.                                                         |
| L - fighter1_Stance    | Fighting stance of fighter 1 (e.g., Orthodox, Southpaw).                    |
| M - fighter1_DOB       | Date of birth of fighter 1.                                                 |
| N - fighter1_SLpM      | Significant strikes landed per minute by fighter 1.                         |
| O - fighter1_StrAcc    | Striking accuracy of fighter 1.                                             |
| P - fighter1_SApM      | Significant strikes absorbed per minute by fighter 1.                       |
| Q - fighter1_StrDef    | Striking defense percentage of fighter 1.                                   |
| R - fighter1_TDAvg     | Average takedowns per 15 minutes for fighter 1.                             |
| S - fighter1_TDAcc     | Takedown accuracy of fighter 1.                                             |
| T - fighter1_TDDef     | Takedown defense percentage of fighter 1.                                   |
| U - fighter1_SubAvg    | Average number of submissions attempted per 15 minutes by fighter 1.        |
| V - fighter2_Name      | Full name of fighter 2.                                                     |
| W - fighter2_Nickname  | Nickname of fighter 2.                                                      |
| X - fighter2_Record    | Fight record (wins-losses-draws) of fighter 2.                              |
| Y - fighter2_Height    | Height of fighter 2.                                                        |
| Z - fighter2_Weight    | Weight of fighter 2.                                                        |
| AA - fighter2_Reach    | Reach of fighter 2.                                                         |
| AB - fighter2_Stance   | Fighting stance of fighter 2 (e.g., Orthodox, Southpaw).                    |
| AC - fighter2_DOB      | Date of birth of fighter 2.                                                 |
| AD - fighter2_SLpM     | Significant strikes landed per minute by fighter 2.                         |
| AE - fighter2_StrAcc   | Striking accuracy of fighter 2.                                             |
| AF - fighter2_SApM     | Significant strikes absorbed per minute by fighter 2.                       |
| AG - fighter2_StrDef   | Striking defense percentage of fighter 2.                                   |
| AH - fighter2_TDAvg    | Average takedowns per 15 minutes for fighter 2.                             |
| AI - fighter2_TDAcc    | Takedown accuracy of fighter 2.                                             |
| AJ - fighter2_TDDef    | Takedown defense percentage of fighter 2.                                   |
| AK - fighter2_SubAvg   | Average number of submissions attempted per 15 minutes by fighter 2.        |


# UFC Fight Outcome Prediction - Presentation

## Problem Definition
- Binary classification problem predicting UFC fight outcomes
- Target variable: Fighter 1 win (1) vs Fighter 2 win (0)
- Features: Differences in physical attributes, fighting records and performance metrics
- Dataset: 2,000 UFC fight records with comprehensive fighter statistics

## Exploratory Data Analysis
- Well-balanced classes (~50% win rate for each fighter position)
- Strong correlation between win differential and fight outcomes (r = 0.67)
- Moderate correlation for striking metrics (SLpM, StrAcc, StrDef)
- Weaker but significant correlation for physical attributes (weight, reach)
- Feature distributions show clear separation for key metrics

## Methodology
- Data preprocessing: Feature engineering, scaling, train-test split (80/20)
- Feature engineering: Created differential features between fighters
- Model selection: Decision Tree, Random Forest, SVM, Neural Network, KNN
- Hyperparameter tuning: GridSearchCV with 5-fold cross-validation
- Evaluation metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC

## Results
- Random Forest achieved highest accuracy (87.5%) and F1 score (0.874)
- Top predictive features:
  1. Win differential (coef: 0.716, odds ratio: 2.046)
  2. Strikes Landed per Minute differential (coef: 0.366, odds ratio: 1.442)
  3. Striking Accuracy differential (coef: 0.116, odds ratio: 1.123)
  4. Weight differential (coef: 0.081, odds ratio: 1.084)
  5. Striking Defense differential (coef: 0.085, odds ratio: 1.089)

## Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.875 | 0.882 | 0.867 | 0.874 |
| Neural Network | 0.845 | 0.858 | 0.828 | 0.843 |
| SVM | 0.835 | 0.842 | 0.825 | 0.833 |
| Decision Tree | 0.795 | 0.784 | 0.813 | 0.798 |
| KNN | 0.790 | 0.803 | 0.771 | 0.787 |

## Analysis and Insights
- Win history is the strongest predictor of fight outcomes
- Striking metrics collectively have significant predictive power
- Stance differences show minimal impact on prediction
- Feature importance aligns with domain expertise
- Random Forest provides best balance of performance and generalization

## Conclusion
- Successfully predicted UFC fight outcomes with high accuracy (87.5%)
- Identified key predictive features that align with provided significance data
- Ensemble methods (Random Forest) outperform single-model approaches
- Results confirm the importance of win differential, striking metrics, and physical attributes in determining fight outcomes