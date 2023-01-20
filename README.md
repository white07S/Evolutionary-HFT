# Evolutionary-HFT
Classification of Buy or Sell in HFT data with ensemble model of LightGBM and Random Forest.

### Procedure

#### Data Explanation: Feature Columns
* timestamp str, datetime string.
* bid_price float, price of current bid in the market.
* bid_qty float, quantity currently available at the bid price.
* bid_price float, price of current ask in the market.
* ask_qty float, quantity currently available at the ask price.
* trade_price float, last traded price.
* sum_trade_1s float, sum of quantity traded over the last second.
* bid_advance_time float, seconds since bid price last advanced.
* ask_advance_time float, seconds since ask price last advanced.
* last_trade_time float, seconds since last trade.

#### Labels

* _1s_side int
* _3s_side int
* _5s_side int

* **The first event that will occur in the following x seconds is labeled according to its kind, where:**
* 0 -- No price change.
* 1 -- Bid price decreased.
* 2 -- Ask price increased.

#### Preprocessing

* "Data preprocessing: The first step in the machine learning pipeline is to convert the input data into a format that the model can understand. In this case, we need to convert the Python dictionary into a JSON format.

* Data check: Before proceeding with further processing, it is important to check for any missing or null values in the dataset. To do this, we can use the **`check_null()`** function which will check for any missing values in the dataset.

* Missing value handling: After identifying any missing or null values in the dataset, the next step is to handle them.The **`fill_null()`** function can be used to fill in the missing values based on certain assumptions or logic. In this example, it is stated that the missing values in the 'sum_trade_1s' column are likely to be 0 when the 'last_trade_time' is larger than 1 sec. Therefore, the assumption is made that all missing values in the 'sum_trade_1s' column can be filled with 0. Additionally, the 'last_trade_time' column can also be filled with the previous record's 'last_trade_time' plus a time movement if the record interval is smaller than 1 sec."

#### Feature Engineering

* "Correlation filter: To reduce data redundancy and improve the efficiency of the model, it is important to remove columns that are highly correlated. The **`correlation_filter.filter()`** function can be used to identify and remove any highly correlated columns in the dataset.

* Logical feature engineering: To improve the performance of the model, it is important to create new features that capture the underlying trading logic. The **`feature_eng.basic_features()`** function can be used to create new features based on trading logic.

* Time-rolling feature engineering: In time-series data, it is important to create features that capture the temporal dependencies between observations. The **`feature_eng.lag_rolling_features()`** function can be used to create new features by lagging and rolling the time-series data. This function can help to capture the temporal dependencies and improve the performance of the model."

#### Feature Selection
* "Feature Selection: To improve the performance of the model, it is important to select the most relevant features from the dataset. The **`feature_selection.select()`** function uses a hybrid approach of genetic algorithm selection and feature importance selection to select the most relevant features.

* Genetic algorithm selection: **`feature_selection.GA_features()`** function uses genetic algorithm to select features that maximizes the model's performance.

* Feature importance selection: **`feature_selection.rf_imp_features()`** function uses feature importance scores from random forest to select relevant features.

#### Modelling
* Modelling: To build the model, an ensemble of lightGBM and random forest models are used.

* Random Forest: **`model.random_forest()`** function is used to train a random forest model.

* lightGBM: **`model.lightgbm()`** function is used to train a lightGBM model.

#### Parameter Tuning

* Parameter Tuning: To improve the performance of the model, it is important to fine-tune the model's parameters. Based on the search space, it is decided whether to use grid search or genetic search for lightGBM model's parameter tuning.

* Grid search: **`model.lightgbm()`**model.GS_tune_lgbm() function uses grid search to tune the lightGBM model's parameters.

* Genetic search: **`model.GA_tune_lgbm()`** function uses genetic search to tune the lightGBM model's parameters."