def loss_function(x_axis, y_axis, parameter):
    for i in range(len(x_axis)):
        loss[i] = (parameter * x_axis[i] - y_axis[i]) ** 2
    return loss


def cost_function(loss):
    cost = sum(loss) / len(loss)
    return cost


def update_parameter(x_axis, y_axis, parameter, learning_rate):
    summation_term = 0
    for i in range(len(x_axis)):
        summation_term += x_axis[i] * (parameter * x_axis[i] - y_axis[i])
    update_value = (2 / len(x_axis)) * summation_term
    parameter -= learning_rate * summation_term
    return parameter


x_axis = [1, 2, 1.5, 1.6, 3, 2.6, 5, 4.1, 4, 4.8]
y_axis = [1, 2, 0.7, 2.4, 3, 2.5, 5, 3.5, 4.8, 4.3]
parameter = 0.1
loss = [0] * len(x_axis)
learning_rate = 0.001
epochs = 1000

for i in range(epochs):
    loss = loss_function(x_axis, y_axis, parameter)
    print("The cost is:")
    print(cost_function(loss))
    parameter = update_parameter(x_axis, y_axis, parameter, learning_rate)
    print("The new value of the parameter is:")
    print(parameter)
