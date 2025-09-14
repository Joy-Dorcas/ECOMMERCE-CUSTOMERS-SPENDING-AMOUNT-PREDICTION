# ECOMMERCE-CUSTOMERS-SPENDING-AMOUNT-PREDICTION
# E-commerce Customer Spending Amount Prediction

A machine learning project that uses linear regression to predict yearly customer spending based on user behavior and engagement metrics.

## 📊 Dataset Overview

The dataset contains **500 customer records** with the following features:

| Column | Type | Description |
|--------|------|-------------|
| Email | object | Customer email address (identifier) |
| Address | object | Customer address information |
| Avatar | object | Customer avatar/profile information |
| Avg. Session Length | float64 | Average time spent per session (minutes) |
| Time on App | float64 | Total time spent on mobile app (minutes) |
| Time on Website | float64 | Total time spent on website (minutes) |
| Length of Membership | float64 | Duration of customer membership (years) |
| **Yearly Amount Spent** | float64 | **Target variable** - Annual spending amount ($) |

**Dataset Size:** 500 entries × 8 columns (31.4+ KB)  
**Missing Values:** None (all columns have 500 non-null values)

## 🎯 Project Objective

Predict the **Yearly Amount Spent** by customers using their behavioral patterns and engagement metrics through linear regression analysis.

## 🔍 Key Features for Prediction

- **Avg. Session Length**: Customer engagement per session
- **Time on App**: Mobile app usage patterns
- **Time on Website**: Web platform engagement
- **Length of Membership**: Customer loyalty indicator

## 🛠️ Technologies Used

- **Python** - Programming language
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning library
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **Jupyter Notebook** - Development environment

## 📁 Project Structure

```
ecommerce-spending-prediction/
├── data/
│   └── ecommerce_customers.csv
├── notebooks/
│   └── customer_spending_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── prediction.py
├── models/
│   └── linear_regression_model.pkl
├── results/
│   ├── model_performance.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/ecommerce-spending-prediction.git
cd ecommerce-spending-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook notebooks/customer_spending_analysis.ipynb
```

## 📈 Model Performance

The linear regression model analyzes the relationship between customer behavior metrics and spending patterns to provide accurate predictions for business decision-making.

### Key Metrics
- **R² Score**: Model accuracy
- **Mean Absolute Error (MAE)**: Average prediction error
- **Root Mean Square Error (RMSE)**: Prediction variance

## 🔮 Usage

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load and prepare data
df = pd.read_csv('data/ecommerce_customers.csv')

# Select features for prediction
features = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
X = df[features]
y = df['Yearly Amount Spent']

# Train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```

## 📊 Business Applications

- **Customer Segmentation**: Identify high-value customers
- **Marketing Strategy**: Target customers with higher spending potential
- **Resource Allocation**: Focus on platforms (app vs website) that drive more spending
- **Customer Retention**: Understand the impact of membership length on spending

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - [GitHub Profile](https://github.com/your-username)

## 🙏 Acknowledgments

- Dataset source and inspiration
- Scikit-learn documentation and community
- Machine learning best practices resources
