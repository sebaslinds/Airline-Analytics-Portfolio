# Flight Delay Prediction Using Machine Learning

This project aims to predict flight delays using real-world U.S. domestic flight data from 2015. The analysis explores key delay factors, performs feature engineering, and builds predictive models (Logistic Regression, Decision Tree, and Random Forest) to support better operational decisions in aviation.

---

## Project Structure

 flight-delay-prediction/
├── data/ # Source & cleaned data files
├── notebooks/ # Jupyter notebooks with analysis & modeling
├── images/ # Visualizations & plots
├── README.md # Project documentation
└── requirements.txt # Dependencies

---

## Objectives

- Perform EDA to understand delay patterns.
- Quantify relationships between categorical features and delays using **Cramér's V**.
- Explore correlations among numeric variables using **heatmaps**.
- Build classification models to predict if a flight will be delayed (binary classification).
- Evaluate model performance using metrics like **accuracy, precision, recall, F1-score, ROC AUC**, and **confusion matrices**.

---

## Data

The dataset includes over 5 million U.S. domestic flights from 2015 with the following features:

- **Categorical:** `AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `DAY_OF_WEEK`
- **Numerical:** `SCHEDULED_DEPARTURE`, `DEPARTURE_DELAY`, `SCHEDULED_ARRIVAL`, `ARRIVAL_DELAY`, `AIR_TIME`, `DISTANCE`

---

## Exploratory Data Analysis

- **Delay Distribution:** Most delays are under 60 minutes; KDE plots show distinct density curves between delayed and on-time flights.
- **Correlation Heatmap:** Strong correlation between `DEPARTURE_DELAY` and `ARRIVAL_DELAY`, as well as between `AIR_TIME` and `DISTANCE`.
- **Cramér’s V Analysis:**
  - `ORIGIN_AIRPORT`: **0.092**
  - `DESTINATION_AIRPORT`: **0.085**
  - `AIRLINE`: **0.084**
  - `DAY_OF_WEEK`: **0.030**

---

## Model Training & Evaluation

### Logistic Regression
- Accuracy: **93%**
- AUC: **0.93**
- Balanced performance on both classes

### Random Forest (Untuned)
- AUC: **0.91**
- Failed to predict delays (likely due to class imbalance or overfitting)

### Decision Tree
- AUC: **0.82**
- Good interpretability, but lower performance

### Evaluation Plots
- Precision-Recall Curves
- ROC Curves
- Confusion Matrices
- Classification Reports

---

## Recommendations

- Use **logistic regression** as the baseline production model.
- Tune **random forest** hyperparameters for improved recall.
- Integrate **weather** and **traffic** data to enhance feature richness.
- Prioritize flight monitoring by **origin/destination airport** risk profiles.

---

## Dependencies

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy


