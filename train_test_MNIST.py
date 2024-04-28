import os
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from RBM import RBM
from au import AutoEncoder
from datetime import datetime


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("MNIST", one_hot=True)

# Train Test splits
X_train, y_train, X_test, y_test = (
    mnist_data.train.images,
    mnist_data.train.labels,
    mnist_data.test.images,
    mnist_data.test.labels,
)

# Scale the data matrix to [0, 1] range
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.fit_transform(X_test)

# Define Network params
nb_epoch = 1
batch_size = 30
iters = len(X_train) / batch_size

# RBMs
rbm_1 = RBM(784, 1000, ["rbmW_1", "rbma_1", "rbmb_1"], 0.3)
rbm_2 = RBM(1000, 500, ["rbmW_2", "rbma_2", "rbmb_2"], 0.3)
rbm_3 = RBM(500, 250, ["rbmW_3", "rbma_3", "rbmb_3"], 0.3)
rbm_4 = RBM(250, 30, ["rbmW_4", "rbma_4", "rbmb_4"], 0.3)

# Uncomment below to load pre-trained weights
# rbm_1.restore_weights('./weights/rbmW_1.weights')
# rbm_2.restore_weights('./weights/rbmW_2.weights')
# rbm_3.restore_weights('./weights/rbmW_3.weights')
# rbm_4.restore_weights('./weights/rbmW_4.weights')

# Autoencoder
autoencoder = AutoEncoder(
    784,
    [1000, 500, 250, 30],
    [
        ["rbmW_1", "rbmb_1"],
        ["rbmW_2", "rbmb_2"],
        ["rbmW_3", "rbmb_3"],
        ["rbmW_4", "rbmb_4"],
    ],
    symmetric_weights=True,
)


# Make a directory to store weights
if not os.path.exists("weights"):
    os.makedirs("weights")

# Train first RBM
print("Training first RBM")
start_time = datetime.now()

for i in range(nb_epoch):
    for j in range(iters):
        batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
        rbm_1.partial_fit(batch_xs)
    print("Epoch:", i, "Cost:", rbm_1.compute_cost(X_train))

stop_time = datetime.now()
elapsed_time = stop_time - start_time
print("RBM 1 -- Elapsed Time: ", elapsed_time)
print("Saving Weights...")
rbm_1.save_weights("./weights/rbm_1.weights")

# Train second RBM
start_time = datetime.now()
print("Training second RBM")
for i in range(nb_epoch):
    for j in range(iters):
        batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
        batch_xs = rbm_1.transform(batch_xs)
        rbm_2.partial_fit(batch_xs)
    print("Epoch:", i, "Cost:", rbm_2.compute_cost(rbm_1.transform(X_train)))

stop_time = datetime.now()
elapsed_time = stop_time - start_time
print("RBM 2 -- Elapsed Time: ", elapsed_time)
print("Saving Weights...")
rbm_2.save_weights("./weights/rbm_2.weights")

# Train third RBM
start_time = datetime.now()
print("Training third RBM")
for i in range(nb_epoch):
    for j in range(iters):
        batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
        batch_xs = rbm_2.transform(rbm_1.transform(batch_xs))
        rbm_3.partial_fit(batch_xs)
    print(
        "Epoch:",
        i,
        "Cost: ",
        rbm_3.compute_cost(rbm_2.transform(rbm_1.transform(X_train))),
    )

stop_time = datetime.now()
elapsed_time = stop_time - start_time
print("RBM 3 -- Elapsed Time: ", elapsed_time)
print("Saving Weights...")
rbm_3.save_weights("./weights/rbm_3.weights")


# Train fourth RBM
start_time = datetime.now()
print("Training fourth RBM")
for i in range(nb_epoch):
    for j in range(iters):
        batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
        batch_xs = rbm_3.transform(rbm_2.transform(rbm_1.transform(batch_xs)))
        rbm_4.partial_fit(batch_xs)
    print(
        "Epoch:",
        i,
        "Cost:",
        rbm_4.compute_cost(rbm_3.transform(rbm_2.transform(rbm_1.transform(X_train)))),
    )

stop_time = datetime.now()
elapsed_time = stop_time - start_time
print("RBM 4 -- Elapsed Time: ", elapsed_time)
print("Saving Weights...")
rbm_4.save_weights("./weights/rbm_4.weights")


# Load RBM weights to Autoencoder
autoencoder.load_rbm_weights("./weights/rbm_1.weights", ["rbmW_1", "rbmb_1"], 0)
autoencoder.load_rbm_weights("./weights/rbm_2.weights", ["rbmW_2", "rbmb_2"], 1)
autoencoder.load_rbm_weights("./weights/rbm_3.weights", ["rbmW_3", "rbmb_3"], 2)
autoencoder.load_rbm_weights("./weights/rbm_4.weights", ["rbmW_4", "rbmb_4"], 3)

# Uncomment below to load pre-trained weights
# autoencoder.load_weights('./weights/ae.weights')

# Train Autoencoder
print("Training Autoencoder")
start_time = datetime.now()
for i in range(nb_epoch):
    cost = 0.0
    for j in range(iters):
        batch_xs, batch_ys = mnist_data.train.next_batch(batch_size)
        cost += autoencoder.partial_fit(batch_xs)
    print("Epoch:", i, "Cost:", cost)

stop_time = datetime.now()
elapsed_time = stop_time - start_time
print("AutoEnocder -- Elapsed Time: ", elapsed_time)
print("Saving Weights...")

autoencoder.save_weights("./weights/ae.weights")
autoencoder.load_weights("./weights/ae.weights")

fig, ax = plt.subplots()

print(autoencoder.transform(X_test)[:, 0])
print(autoencoder.transform(X_test)[:, 1])

plt.scatter(
    autoencoder.transform(X_test)[:, 0], autoencoder.transform(X_test)[:, 1], alpha=0.5
)
plt.show()
plt.savefig("ae_scatter_plot")
