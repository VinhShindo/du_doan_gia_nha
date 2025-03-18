from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

data = pd.read_csv('Data_Set_PriceHome.csv')

def linear_activation(x):  
    return x

def linear_activation_derivative(x):  
    return np.ones_like(x)

def initialize_neural_network(layers):  
    W, b = [], []
    for i in range(len(layers) - 1):
        W.append(np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i]))
        b.append(np.zeros((layers[i + 1], 1)))
    return W, b  

def forward_propagation(X, W, b, layers):
    A = [X]
    out = A[-1]
    for i in range(len(layers) - 1):
        out = linear_activation(np.dot(out, W[i]) + b[i].T)
        A.append(out)
    return A  

def backpropagation(X, y, A, W, b, layers, alpha, lambd):
    y = y.reshape(-1, 1)
    m = y.shape[0]
    dA = [A[-1] - y]
    dW, db = [], []

    for i in reversed(range(len(layers) - 1)):
        dA_hat = dA[-1] * linear_activation_derivative(A[i + 1])
        dw_ = np.dot(A[i].T, dA_hat) / m + (lambd / m) * W[i]
        db_ = np.sum(dA_hat, axis=0, keepdims=True) / m
        dA_ = np.dot(dA_hat, W[i].T)

        dW.append(dw_)
        db.append(db_)
        dA.append(dA_)

    dW = dW[::-1]
    db = db[::-1]

    for i in range(len(layers) - 1):
        W[i] -= alpha * dW[i]
        b[i] -= alpha * db[i].T

    return W, b

def calculate_loss(y_true, y_predict, W, lambd):
    m = y_true.shape[0]
    loss = np.mean((y_predict - y_true) ** 2)
    reg_loss = (lambd / (2 * m)) * sum(np.linalg.norm(w) ** 2 for w in W)
    return loss + reg_loss  

def train_neural_network(X, y, layers, alpha, epochs, verbose, lambd):
    W, b = initialize_neural_network(layers)

    for epoch in range(epochs):
        A = forward_propagation(X, W, b, layers)
        y_predict = A[-1]
        
        W, b = backpropagation(X, y, A, W, b, layers, alpha, lambd)

        loss = calculate_loss(y, y_predict, W, lambd)

        if epoch % verbose == 0:
            print(f"Epoch {epoch}, Loss: {loss:.5f}")

    return W, b  

def predict(X, W, b, layers):
    for i in range(len(layers) - 1):
        X = linear_activation(np.dot(X, W[i]) + b[i].T)
    return X  

def inverse_scale(y_scaled, scaler):
    return scaler.inverse_transform(y_scaled)

def train_model():
    N, d = data.shape
    X = data.iloc[:, 0:d-1].values.astype(float)
    y = data.iloc[:, d-1].values.reshape(-1, 1).astype(float)

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    layers = [X.shape[1], 64, 1]

    W, b = train_neural_network(X, y_scaled, layers, alpha=0.001, epochs=500, verbose=10, lambd=0.00001)

    return W, b, scaler_X, scaler_y

@app.route('/', methods=['GET', 'POST'])
def index():
    data_out = data.values.astype(float).tolist()

    W, b, scaler_X, scaler_y = train_model()

    if request.method == 'POST':
        year_built = int(request.form['year_built'])
        age_of_house = int(request.form['age_of_house'])
        distance_to_city_center = float(request.form['distance_to_city_center'])
        num_stores = int(request.form['num_stores'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
    
        input_data = np.array([[year_built, age_of_house, distance_to_city_center, num_stores, latitude, longitude]])
        input_data_scaler = scaler_X.transform(input_data)

        predictions_scaled = predict(input_data_scaler, W, b, [input_data_scaler.shape[1], 64, 1])
        predictions = inverse_scale(predictions_scaled, scaler_y)

        avg_price = data[data['Year of sale of the house'] == year_built]['Price of houses'].mean()
        nearby_houses = data[(data['Distance from city center'] >= distance_to_city_center - 1) & 
                             (data['Distance from city center'] <= distance_to_city_center + 1)]
        avg_area_price = nearby_houses['Price of houses'].mean()
        price_difference = predictions.flatten()[0] - avg_price
        data_out = data.values.astype(float).tolist()

        predicted_price = predictions.flatten()[0]
        predicted_price_vnd = "{:,.2f}".format(round(predicted_price, 2))
        avg_price = "{:,.2f}".format(avg_price.flatten()[0])
        avg_area_price = "{:,.2f}".format(avg_area_price.flatten()[0])
        price_difference = "{:,.2f}".format(abs(price_difference).flatten()[0])

        result = {
            'predicted_price_vnd': predicted_price_vnd,
            'avg_price': avg_price,
            'avg_area_price': avg_area_price,
            'price_difference': price_difference,
            'error': None,
            'input_data': {
                'year_built': year_built,
                'age_of_house': age_of_house,
                'distance_to_city_center': distance_to_city_center,
                'num_stores': num_stores,
                'latitude': latitude,
                'longitude': longitude
            }
        }
        return jsonify(result)

    return render_template('index.html', data_out=data_out)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
