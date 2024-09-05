# Diabetes Checkup

![Diabetes Checkup](https://user-images.githubusercontent.com/yourusername/diabetes-checkup-banner.png)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Model Details](#model-details)
- [Visualizations](#visualizations)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

**Diabetes Checkup** is an interactive web application built with Streamlit that allows users to assess their risk of diabetes based on various health parameters. Leveraging machine learning, the app provides predictions and visual insights to help users understand their health status.

## Features

- **Interactive User Input:** Users can input their health data through intuitive sliders.
- **Real-time Prediction:** Utilizes a Random Forest Classifier to predict diabetes risk.
- **Visual Insights:** Multiple scatter plots compare user data against the training dataset.
- **Data Statistics:** Provides descriptive statistics of the training dataset for context.
- **Model Accuracy:** Displays the accuracy of the predictive model.

## Installation

Follow these steps to set up and run the Diabetes Checkup application on your local machine.

### Prerequisites

- Python 3.7 or higher installed on your system.
- `pip` package manager.

### Clone the Repository

```bash
git clone(https://github.com/swapnil77122/Diabetes-Diseasen-Prediction-using-streamlit.git)
```

### Install Dependencies

It's recommended to use a virtual environment to manage dependencies.

#### Using `venv`

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Required Packages

You can install the required packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Alternatively, install the packages individually:

```bash
pip install streamlit pandas scikit-learn numpy matplotlib plotly seaborn pillow
```


## Usage

Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

Replace `app.py` with the name of your Python script if it's different.

Once the app is running, it will open in your default web browser. Use the sidebar to input your health data and view the prediction along with various visualizations.

## Project Structure

```
diabetes-checkup/
│
├── app.py                 # Main Streamlit application
├── diabetes.csv           # Dataset file
├── requirements.txt       # Python dependencies
├── README.md              # This README file
└── images/                # Directory for images and screenshots
```

## Dependencies

The project relies on the following Python libraries:

- **Streamlit:** For building the web application.
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical operations.
- **Scikit-learn:** For machine learning algorithms and metrics.
- **Matplotlib & Seaborn:** For data visualization.
- **Plotly:** For interactive plots.
- **Pillow (PIL):** For image processing.

### Installing Dependencies

You can install all dependencies using `pip`:

```bash
pip install streamlit pandas scikit-learn numpy matplotlib plotly seaborn pillow
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Model Details

The application uses a **Random Forest Classifier** to predict the likelihood of diabetes. The dataset is split into training and testing sets with an 80-20 ratio. After training, the model's accuracy is displayed to give users an understanding of its performance.

### Model Training

```python
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
```

### Accuracy

The model's accuracy is calculated using the `accuracy_score` metric from Scikit-learn and displayed as a percentage.

```python
accuracy_score(y_test, rf.predict(x_test))*100
```

## Visualizations

The app provides several scatter plots to help users visualize their data in comparison to the training dataset:

1. **Pregnancy Count vs Age**
2. **Glucose Levels vs Age**
3. **Blood Pressure vs Age**
4. **Skin Thickness vs Age**
5. **Insulin Levels vs Age**
6. **BMI vs Age**
7. **Diabetes Pedigree Function vs Age**

Each plot highlights the user's input data point in either blue (healthy) or red (diabetic) based on the prediction.

### Example Visualization

![Pregnancy Count Graph](https://user-images.githubusercontent.com/yourusername/pregnancy-count-graph.png)

*Pregnancy Count Graph showing user's data point compared to others.*

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the excellent framework.
- [Kaggle](https://www.kaggle.com/) for providing the Pima Indians Diabetes Dataset.
- The open-source community for the various libraries and tools used in this project.

---


