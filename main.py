import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import math

#####################################################################################
def Ring(big_radius, small_radius, centerX, centerY):
    x = 0
    y = 0

    while abs(x) < small_radius and abs(y) < small_radius:
        r = big_radius * math.sqrt(random())
        theta = random() * 2 * math.pi
        x = centerX + r * math.cos(theta)
        y = centerY + r * math.sin(theta)

    return np.array([x, y])


def RingArray(big_radius, small_radius, points_amount, centerX, centerY):
    X = []
    counter = 0
    while counter < points_amount:
        x = np.random.random()*2*np.sqrt(2) - np.sqrt(2)
        y = np.random.random()*2*np.sqrt(2) - np.sqrt(2)
        if 1<= x**2 + y**2 <= 2:
            X.append((x, y))
            counter+=1
    X = np.array(X)
    return X

def NeuronsInLine(neurons_size, radius):
    line = np.array([(i , 1-i) for i in np.linspace(1,0, num= neurons_size)]).reshape(neurons_size,2)
    return line


def NeuronsInCircle(neurons_size, radius):
    neurons = np.array([(i, i+0.001) for i in np.linspace(-1.5,1.5, num= neurons_size)]).reshape(neurons_size,2)
    return neurons

def getCoordinates(array, type):
        x = array[:, 0]
        y = array[:, 1]
        return x, y


def SquareNonUniform(Min, Max, sample_size):
    data = np.random.normal(0, 1, size=(sample_size, 2))
    # data = np.random.geometric(p=0.35, size=(sample_size,2))
    min = np.min(data)
    max = np.max(data)
    data = (data -min)/(max-min)
    return data


def SquareUniform(Min, Max, samples_size):
    data = np.random.uniform(0, 1, size=(samples_size, 2))
    return data

def RandomPointsHand(samples_size):
    data = []
    mask = cv2.cvtColor(cv2.imread('Hand.jpeg'), cv2.COLOR_BGR2GRAY) / 255.0
    while (len(data) < samples_size):
        x = np.random.random()
        y = np.random.random()
        if (mask[int(y * 1000), int(x * 1000)] == 0):
            data.append((x, -y))
    return np.array(data)

#######################################################################################

def euclideanDistance(point, neuron):
    return math.sqrt(math.pow(point[0] - neuron[0], 2) + math.pow(point[1] - neuron[1], 2))


# choose nearest neuron to current point
def chooseNeuronToMove(point, neurons):
    min_distance = math.inf
    min_distance_index = -1
    for index in range(len(neurons) - 1):
        distance = euclideanDistance(point, neurons[index])
        if distance < min_distance:
            min_distance = distance
            min_distance_index = index

    return min_distance_index


# topological neighborhood
def neighborFunction(winner_neuron, neuron, sigma):
    distance = euclideanDistance(winner_neuron, neuron)
    return math.exp(- (math.pow(distance, 2) / (2 * math.pow(sigma, 2))))


# update neuron place
def updateNeuronPlacement(point, neuron, alpha, h):
    return neuron + alpha * h * (point - neuron)


# update sigma by iteration number
def updateSigma(sigma, iteration, lambda_start):
    # return sigma / (1 + (current_iteration / iteration_size))
    return sigma * math.exp(-iteration / lambda_start)


# update alpha value
def updateAlpha(alpha, iteration, lambda_start):
    return alpha * math.exp(-iteration / lambda_start)


# start algorithm
def fit(points, neurons, epochs, radius):
    alpha_start = 0.01

    sigma_start = radius / 2 + 0.0001

    lambda_start = epochs / math.log(epochs)

    errors = np.array([])

    # train each point by epochs amount
    for iteration in range(epochs):
        # update learning variables
        alpha = updateAlpha(alpha_start, iteration, epochs)
        sigma = updateSigma(sigma_start, iteration, lambda_start)

        sum_of_errors = 0

        # iterate on the data points
        for point in points:

            # find winner
            winner_neuron_index = chooseNeuronToMove(point, neurons)

            # calculate error
            sum_of_errors += euclideanDistance(point, neurons[winner_neuron_index])

            # for each neuron
            for neighbor_index in range(len(neurons)):

                distance = euclideanDistance(neurons[winner_neuron_index], neurons[neighbor_index])

                if distance < sigma:
                    h = neighborFunction(neurons[winner_neuron_index], neurons[neighbor_index], sigma)
                    neurons[neighbor_index] = updateNeuronPlacement(point, neurons[neighbor_index], alpha, h)

        errors = np.append(errors, [sum_of_errors / len(points)])

    return errors
#################################################################

radius = 0.2
centerX = -1/2
centerY = -1/2
neurons_size = 30

# type of train shape -> ["circle", "ring"]
type_of_shape = "squareNuniform"

# create random points by chosen type of shape
if type_of_shape == "ring":
    X = RingArray(radius * 2, radius, 300, centerX, centerY)
elif type_of_shape == "squareNuniform":
    X = SquareNonUniform(0, 1, 2000)
elif type_of_shape == "squareUniform":
    X = SquareUniform(0, 1, 300)


# type of neurons network shape -> ["line", "circle", "square"]
type_of_network = "line"

# create neurons points by chosen type of network
if type_of_network == "line":
    neurons = NeuronsInLine(neurons_size, radius)
elif type_of_network == "circle":
    neurons = NeuronsInCircle(neurons_size, radius)

# algorithm
epochs = 300
errors = fit(X, neurons, epochs, radius)

# get the coordinates of neurons for presentation on the graph
x_values, y_values = getCoordinates(neurons, type_of_network)

# paint points and neurons
fig, ax = plt.subplots()
plt.scatter(X[:, 0], X[:, 1], color='black', label='points')
plt.scatter(neurons[:, 0], neurons[:, 1], color='red', marker='o', label='neurons')

if type_of_shape == "ring":
    c1 = plt.Circle((0, 0), 1, color='blue', fill=False)
    c2 = plt.Circle((0, 0), np.sqrt(2), color='blue', fill=False)
    ax.add_artist(c1)
    ax.add_artist(c2)
plt.show()

plt.plot(range(1, len(errors) + 1), errors, color='red')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()

