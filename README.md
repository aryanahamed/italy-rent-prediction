# Italy Rent Prediction
This hobby project estimates advertised monthly rents for Italian properties from an archived listing snapshot.

## Installation
To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/aryanahamed/italy-rent-prediction.git
cd italy-rent-prediction
pip install -r requirements.txt
```
Alternatively, make changes on the notebook directly after cloning.

## Data Collection
The deployed model uses the [Italy house-prices dataset on Kaggle](https://www.kaggle.com/datasets/tommasoramella/italy-house-prices/data). The app currently describes this source as data through 2023-12-07. It is not a live rent feed or a longitudinal market dataset.

## Usage
Check the following page to try out the model.

[Italy Rent Prediction](https://italy-rent-prediction.streamlit.app/)

### Running Locally
To run the Streamlit app locally:

```bash
streamlit run Home.py
```

## Technical Details

### Model
- **Algorithm**: XGBoost Regressor
- **Artifact/runtime contract**: serialized with XGBoost 3.1.1; runtime constrained to `>=3.1.1,<4.0.0`
- **Features**: 36 model inputs including location, size, amenities, condition, and derived ratios
- **Target**: Log-transformed rent price (to handle skewness)
- **Derived ratios**: `rooms / (area + 1)` and `bathrooms / (rooms + 1)`, matching training

### Input Stability Diagnostic

The stability score varies supported property inputs locally: area by ±5% and rooms/bathrooms by one within the app's bounds. Derived ratios are recomputed for every scenario. The displayed range and 0–100 score describe local model sensitivity only; they are not a confidence interval, accuracy estimate, or probability statement.

### One-Feature Sensitivity

Binary and encoded-category flags are compared with a stated zero reference while the other inputs stay fixed. Continuous and derived features are omitted unless a defensible training reference is available. These checks are model associations, not causal effects, and do not form an additive price breakdown.

### Current Data Limitations

- Historical price trends are unavailable because a listing snapshot cannot establish change over time.
- Geographic rental maps are temporarily unavailable because the legacy coordinate cache is keyed only by neighborhood name and can merge places that share a name.
- Comparable records are restricted to the selected city and matched only on rooms and area; the model estimate is not used to choose them.
- Affordability is calculated only when the user supplies monthly disposable household income. The 30% ratio is presented as a reference, not personalized financial advice.


## Sample Plots
[![Heatmap](https://i.ibb.co/Lz79PC8/output.png)](https://ibb.co/kgCqyhx)
[![Rent Price vs Area](https://i.ibb.co/C1wh61T/output2.png)](https://ibb.co/Tvwm1vF)
[![Flats in each city](https://i.ibb.co/3dmjXLs/newplot.png)](https://ibb.co/4tRXy5f)


## Contributing
Contributions are welcome and appreciated!

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
