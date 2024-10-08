import torch
from torch import nn
import matplotlib.pyplot as plt

# Create Parameters
weight = 0.3
bias = 0.9

# Create List of Features and Labels (Data)
start = 0
end = 1
step = 0.005
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * x + bias

# Create Train Set and Test Set
train_split = int(len(x) * 0.8)
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

# Define Plot Predictions


def plot_predictions(train_data=x_train,
                     train_labels=y_train,
                     test_data=x_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})


# plot_predictions()
# plt.show()

# Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(
            1, dtype=float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(
            1, dtype=float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(400)  # Pool where to get random weights and biases
loss_fn = nn.L1Loss()  # Mean Average Error

# Create Initial Model
model_0 = LinearRegressionModel()

optimizer = torch.optim.SGD(
    params=model_0.parameters(), lr=0.005, momentum=0.93)

torch.manual_seed(400)

epochs = 300

epoch_count = []
train_loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()

    with torch.inference_mode():
        test_pred = model_0(x_test)
        test_loss = loss_fn(test_pred, y_test.type(torch.float))
        if epoch % 20 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(
                f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss}")

# Plot the loss curves

# with torch.inference_mode():
#     y_preds = model_0(x_test)

plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plot_predictions(predictions=test_pred)
plt.show()

# Find the Learned Parameters of the Model
print("The model learned the follwing values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the orignial values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")
