# 📈 Google Stock Price Prediction using RNN (LSTM)
This project involves predicting Google’s stock opening prices using Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) layers. The objective was to use past stock market data to build a time-series model that could forecast future stock trends, specifically for January 2017.

# ✅ Step-by-Step Process
1. Data Collection
Used historical GOOG stock price data from a CSV file containing features like Open, High, Low, Close, and Volume.

2. Data Preprocessing
Selected only the ‘Open’ column for modeling.

Applied MinMaxScaler to scale the data between 0 and 1, improving neural network performance.

Created time-series sequences: For each data point at time t, the model uses the previous 60 days of prices (t-60 to t-1) as features.

Converted the resulting lists into NumPy arrays and reshaped them into a 3D format (samples, time steps, features) to fit LSTM input requirements.

3. Model Building (RNN with LSTM)
Constructed a Sequential RNN model using Keras:

4 stacked LSTM layers, each with 50 units.

Dropout layers (20%) after each LSTM to prevent overfitting.

A final Dense layer with 1 neuron to output the predicted price.

Model Summary:

LSTM → Dropout → LSTM → Dropout → LSTM → Dropout → LSTM → Dropout → Dense

4. Model Training
Compiled the model with:

Optimizer: Adam (efficient for time-series problems)

Loss function: Mean Squared Error (MSE)

Trained the model for 100 epochs with a batch size of 32 using the prepared training sequences.

5. Preparing Test Data
Merged the last 60 days of training data with the test data to prepare the input for January 2017 predictions.

Applied the same scaling and sequence formatting as done for training.

6. Prediction & Evaluation
Predicted the stock prices for January 2017 using the trained model.

Inverse transformed the predictions to get the actual price scale.

Compared the predicted values vs real values using a line graph to visualize the performance.

# 📊 Techniques Used
Time-series data generation (sliding window of 60 time steps)

Data normalization using MinMaxScaler

Deep learning with stacked LSTM layers

Dropout regularization to combat overfitting

Model evaluation with Mean Squared Error

Plotting with Matplotlib to visualize model performance

# 🧠 Tools & Libraries
Python, NumPy, Pandas

Keras (TensorFlow backend)

Scikit-learn for preprocessing

Matplotlib for visualization

# 🎯 Final Outcome
Successfully trained an RNN model capable of learning from historical Google stock data.

The predicted stock prices closely followed the actual prices for January 2017, showing the model's ability to capture underlying trends.

# 🙌🏻 Data Scientist 
Behara Pavan Kumar.
