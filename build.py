import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense # type: ignore

# 1. Create sample sequential data
# Example: X = [ [0,1,2], [1,2,3], [2,3,4] ] → Y = [3, 4, 5]

X = []
Y = []

for i in range(100):
    X.append([i, i+1, i+2])
    Y.append(i+3)

X = np.array(X)
Y = np.array(Y)

# Reshape X to be [samples, time_steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# 2. Build the RNN model
model = Sequential()
model.add(SimpleRNN(units=10, activation='tanh', input_shape=(3, 1)))
model.add(Dense(1))

# 3. Compile the model
model.compile(optimizer='adam', loss='mse')

# 4. Train the model
model.fit(X, Y, epochs=20, verbose=1)

# 5. Predict a new value
test_input = np.array([[97, 98, 99]]).reshape((1, 3, 1))
predicted = model.predict(test_input, verbose=0)
print(f"Predicted next number after [97, 98, 99]: {predicted[0][0]:.2f}")