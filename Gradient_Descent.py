import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

x = np.arange(-100, 400, 5)
y = 0.2 * x - 5 + np.random.randint(-18, 20, size=x.shape)
data = pd.DataFrame({
    'x': x,
    'y': y
})

csv_file_path = 'function_data.csv'
data.to_csv(csv_file_path, index=False)

plt.plot(x, y, 'o', label='Noisy Function (y = 0.2x - 5)', color='blue')
plt.legend()
plt.show()

read_data = pd.read_csv(csv_file_path)
print(read_data.head())

x= read_data['x']
y=read_data['y']

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
class GradientDescent:
    def __init__(self, x, y, learning_rate=0.01):
        self.x = np.array(x)
        self.y = np.array(y)

        self.x_mean = np.mean(self.x)
        self.x_std = np.std(self.x)
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)

        self.x_normalized = (self.x - self.x_mean) / self.x_std
        self.y_normalized = (self.y - self.y_mean) / self.y_std

        self.learning_rate = learning_rate
        self.a = np.random.randn()
        self.c = np.random.randn()

    def predict(self, x):
        return self.a * x + self.c

    def compute_cost(self, a=None, c=None):
        if a is None:
            a = self.a
        if c is None:
            c = self.c
        predictions = a * self.x_normalized + c
        return np.mean((predictions - self.y_normalized) ** 2)

    def compute_gradients(self):
        predictions = self.predict(self.x_normalized)
        dJ_da = 2 * np.mean((predictions - self.y_normalized) * self.x_normalized)
        dJ_dc = 2 * np.mean(predictions - self.y_normalized)
        return dJ_da, dJ_dc

    def update_parameters(self, dJ_da, dJ_dc):
        self.a -= self.learning_rate * dJ_da
        self.c -= self.learning_rate * dJ_dc

    def train(self, num_iterations=10000):
        history = []
        for i in range(num_iterations):
            dJ_da, dJ_dc = self.compute_gradients()
            self.update_parameters(dJ_da, dJ_dc)
            cost = self.compute_cost()
            history.append((self.a, self.c, cost))
            if i % 30 == 0:  
                print(f"Iteration {i}: a = {self.a:.4f}, c = {self.c:.4f}, cost = {cost:.4f}")
            if np.isnan(cost) or np.isinf(cost):
                print(f"Training stopped at iteration {i} due to invalid cost")
                break
        return history

    def get_original_parameters(self):
        a_original = self.a * (self.y_std / self.x_std)
        c_original = self.y_mean - a_original * self.x_mean
        return a_original, c_original

gd = GradientDescent(x, y, learning_rate=1)
history = gd.train(num_iterations=350)

a_original, c_original = gd.get_original_parameters()
print(f"Optimized parameters: a = {a_original:.4f}, c = {c_original:.4f}")

a_range = np.linspace(-1, 1, 100)
c_range = np.linspace(-1, 1, 100)
a_mesh, c_mesh = np.meshgrid(a_range, c_range)

cost_mesh = np.zeros_like(a_mesh)
for i in range(a_mesh.shape[0]):
    for j in range(a_mesh.shape[1]):
        cost_mesh[i, j] = gd.compute_cost(a_mesh[i, j], c_mesh[i, j])

a_values, c_values, loss_values = zip(*history)

fig = go.Figure(data=[
    go.Surface(x=a_mesh, y=c_mesh, z=cost_mesh, colorscale='viridis', opacity=0.8),
    go.Scatter3d(x=a_values, y=c_values, z=loss_values, mode='lines', 
                 line=dict(color='red', width=4), name='Gradient Descent Path')
])

fig.update_layout(
    title='Interactive 3D Plot of Cost Function',
    scene=dict(
        xaxis_title='a',
        yaxis_title='c',
        zaxis_title='Cost',
        aspectmode='cube'
    ),
    width=800,
    height=800,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()
fig.write_html("cost_function_3d.html")
print("Interactive 3D plot saved as 'cost_function_3d.html'")

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original Data')
plt.plot(x, a_original * x + c_original, color='red', label='Fitted Line')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Data and Fitted Line')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original Data')
plt.plot(x, a_original * x + c_original, color='red', label='Fitted Line')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Data and Fitted Line')
plt.show()

print(x)
print(y)