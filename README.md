# UEFA Euro Championship Match Outcome Prediction

## Project Overview
This project aims to predict the outcomes of UEFA Euro Championship matches using historical match data. The prediction model is built using the XGBoost classifier, enhanced with various features derived from match statistics and team performance metrics. The project involves data scraping, cleaning, feature engineering, and model training and evaluation.

## Technologies Used
- **Python**
- **BeautifulSoup**
- **Pandas**
- **XGBoost**
- **SMOTE (Synthetic Minority Over-sampling Technique)**
- **Scikit-learn**

## Project Structure
1. **Data Collection and Preprocessing**
2. **Feature Engineering**
3. **Model Training and Evaluation**
4. **Results and Predictions**

### Data Collection and Preprocessing
- Historical match data is scraped from fbref.com.
- BeautifulSoup is used to parse HTML and extract relevant data.
- Data includes match scores, fixtures, and shooting statistics.
- Data is cleaned and converted to appropriate formats using Pandas.

### Feature Engineering
- Rolling averages for various statistics (goals, shots, possession, etc.) are computed.
- Head-to-head statistics between teams are calculated.
- Team experience is quantified based on the number of matches played in different stages of the tournament.
- Match importance is assigned based on the round of the competition.

### Model Training and Evaluation
- The target variable is the match result (win or not win).
- SMOTE is applied to balance the dataset.
- An XGBoost classifier is used for training the model.
- Hyperparameter tuning is performed using RandomizedSearchCV.
- Model performance is evaluated using cross-validation, accuracy, and precision scores.

### Results and Predictions
- Predictions are saved in a CSV file with probabilities and actual results.
- Confusion matrix, accuracy, and precision scores are printed for evaluation.

## Key Files
- **data_collection.py:** Script for scraping and preprocessing match data.
- **prediction_model.py:** Script for feature engineering, model training, and evaluation.
- **matches.csv:** CSV file containing the processed match data.
- **matches_rolling.csv:** CSV file containing match data with rolling averages.
- **predictions_euro_2021.csv:** CSV file containing match predictions and actual results.

## Results Summary
- **Model Accuracy:** 85% 
- **Model Precision:** 87%
- **Confusion Matrix:**

[[True Positive, False Positive],
[False Negative, True Negative]]


