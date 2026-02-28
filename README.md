# Air Quality Data Analysis – Assignment 02

## Overview
This project analyzes air quality data collected from multiple monitoring stations for the year 2025. The goal is to process raw environmental data, perform statistical and machine learning analysis, and visualize insights through a dashboard.

---

## Project Structure

```
6A_23L-2550_Assignment02/
│
├── fetching/          # Data fetching scripts
├── processing/        # Data cleaning and preprocessing
├── modeling/          # PCA and ML models
├── visualization/     # Graphs and analysis plots
├── dashboard/         # Streamlit dashboard
├── utils/             # Helper functions
├── config.py          # Configuration variables
├── main.py            # Main pipeline runner
├── requirements.txt   # Project dependencies
└── README.md
```

---

## Features

- Data fetching from air quality API
- Data preprocessing and cleaning
- PCA analysis
- Health threshold evaluation
- Industrial vs residential classification
- Streamlit dashboard visualization

---

## Installation

Clone the repository:

```
git clone https://github.com/l232550/6A_23L-2550_Assignment02.git
cd 6A_23L-2550_Assignment02
```

Create a virtual environment:

```
python -m venv venv
source venv/Scripts/activate   # Windows
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

Run the main pipeline:

```
python main.py
```

Run the dashboard:

```
streamlit run dashboard/app.py
```

---

## Dataset

- Year: 2025
- Multiple monitoring stations
- Parameters include PM2.5, PM10, NO2, and O3
- Full 12-month coverage

---

## Author

Tooba Nadeem  
6A – Assignment 02