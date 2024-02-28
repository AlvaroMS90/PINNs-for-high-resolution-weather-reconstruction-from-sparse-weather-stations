# Import libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 
print('Using TensorFlow version: ', tf.__version__, ', GPU:', availale_GPUs)
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import numpy as np
import scipy.io
from tensorflow.keras.layers import Input, Activation, Dense
from tensorflow.keras.models import Model
import pandas as pd
import datetime as dt
import tensorflow_addons as tfa

# Extract data from dataset
WS_data = scipy.io.loadmat('Weather_data.mat')

# Convert date to continuous time: from date and time format to seconds
date_0 = WS_data['Date'][0]
date = []
for i in range(0, len(date_0)):
    date = np.append(date, str(date_0[i])[2 : -2])
time_init = dt.datetime(int(date[0][0 : 4]), int(date[0][5 : 7]), int(date[0][8 : 10]), int(date[0][11 : 13]), int(date[0][14 : 16]))
T_nan_index = np.argwhere(pd.isna(date))
date = np.delete(date, T_nan_index[:, 0],  0)
print('Double-check for NaN in time sequence', np.sum(pd.isna(date)))

Seconds = np.zeros((date.shape[0], 1))
for index in range(date.shape[0]):
    Seconds[index, 0] = ((dt.datetime(int(date[index][0 : 4]), int(date[index][5 : 7]), int(date[index][8 : 10]), int(date[index][11 : 13]), int(date[index][14 : 16])) - time_init).total_seconds())
T_WS = Seconds

# Convert to Cartesian coordinates
X_WS = np.array(6378000 * np.sin(np.radians(WS_data['Lon'])))[0]  # Longitude to meters
Y_WS = np.array(6378000 * np.sin(np.radians(WS_data['Lat'])))[0]  # Latitude to meters
Z_WS = np.array(WS_data['Alt'])[0]
Temp_WS = np.array(WS_data['Temperature'])[0]

# Project wind speed and direction into Cartesian coordinates
U_WS = (WS_data['WindSpeed'] * WS_data['WindDirectionX'])[0]
V_WS = (WS_data['WindSpeed'] * WS_data['WindDirectionY'])[0]

# Pressure from mbar to Pa
P_WS = WS_data['Pressure'][0] * 100

# Remove NaN values from time field
X_WS = np.delete(X_WS, T_nan_index[:, 0],  0)
Y_WS = np.delete(Y_WS, T_nan_index[:, 0],  0)
Z_WS = np.delete(Z_WS, T_nan_index[:, 0],  0)
U_WS = np.delete(U_WS, T_nan_index[:, 0],  0)
V_WS = np.delete(V_WS, T_nan_index[:, 0],  0)
P_WS = np.delete(P_WS, T_nan_index[:, 0],  0)
Temp_WS = np.delete(Temp_WS, T_nan_index[:, 0],  0)

# Structure data into matrix: 21 available stations (rows) x measurement every 10 min (column)
T_WS = np.reshape(T_WS, (int(T_WS.shape[0] / 21), 21)).T # There are 21 WS stations in this case
X_WS = np.reshape(X_WS, (T_WS.shape[1], T_WS.shape[0])).T
Y_WS = np.reshape(Y_WS, (T_WS.shape[1], T_WS.shape[0])).T
Z_WS = np.reshape(Z_WS, (T_WS.shape[1], T_WS.shape[0])).T
U_WS = np.reshape(U_WS, (T_WS.shape[1], T_WS.shape[0])).T
V_WS = np.reshape(V_WS, (T_WS.shape[1], T_WS.shape[0])).T
P_WS = np.reshape(P_WS, (T_WS.shape[1], T_WS.shape[0])).T
Temp_WS = np.reshape(Temp_WS, (T_WS.shape[1], T_WS.shape[0])).T
print('Number of weather stations:', T_WS.shape[0])

# Remove NaN from location data
X_nan_index = np.argwhere(np.isnan(X_WS))
T_WS = np.delete(T_WS, X_nan_index[:, 0],  0)
P_WS = np.delete(P_WS, X_nan_index[:, 0],  0)
U_WS = np.delete(U_WS, X_nan_index[:, 0],  0)
V_WS = np.delete(V_WS, X_nan_index[:, 0],  0)
X_WS = np.delete(X_WS, X_nan_index[:, 0],  0)
Y_WS = np.delete(Y_WS, X_nan_index[:, 0],  0)
Z_WS = np.delete(Z_WS, X_nan_index[:, 0],  0)
Temp_WS = np.delete(Temp_WS, X_nan_index[:, 0],  0)
print('Double-check for NaN in location field', np.sum(np.isnan(X_WS)))

# Days selected for reconstruction
n_days = 14 # Change up to a maximum of 14 availsble days
samples =  int(144 * n_days) # Convert selected days to snapshots
T_WS = T_WS[:, : samples] 
X_WS = X_WS[:, : samples]
Y_WS = Y_WS[:, : samples]
Z_WS = Z_WS[:, : samples]
U_WS = U_WS[:, : samples]
V_WS = V_WS[:, : samples]
P_WS = P_WS[:, : samples]
Temp_WS = Temp_WS[:, : samples]

# Sort values in matrix into increasing values of X coordinate
for snap in range(0, T_WS.shape[1]):
    index_sort = np.argsort(X_WS[:, snap])
    T_WS[:, snap] = T_WS[index_sort, snap]
    X_WS[:, snap] = X_WS[index_sort, snap]
    Y_WS[:, snap] = Y_WS[index_sort, snap]
    Z_WS[:, snap] = Z_WS[index_sort, snap]
    U_WS[:, snap] = U_WS[index_sort, snap]
    V_WS[:, snap] = V_WS[index_sort, snap]
    P_WS[:, snap] = P_WS[index_sort, snap]
    Temp_WS[:, snap] = Temp_WS[index_sort, snap]

# Delete NaN from U, V and P if constantly occuring for each weather station
uvp_mean = np.nanmean(np.concatenate([U_WS, V_WS, P_WS], axis = 1), axis = 1)[:, None]
vel_nan_index = np.argwhere(np.isnan(uvp_mean))
T_WS = np.delete(T_WS, vel_nan_index[:, 0],  0)
P_WS = np.delete(P_WS, vel_nan_index[:, 0],  0)
U_WS = np.delete(U_WS, vel_nan_index[:, 0],  0)
V_WS = np.delete(V_WS, vel_nan_index[:, 0],  0)
X_WS = np.delete(X_WS, vel_nan_index[:, 0],  0)
Y_WS = np.delete(Y_WS, vel_nan_index[:, 0],  0)
Z_WS = np.delete(Z_WS, vel_nan_index[:, 0],  0)
Temp_WS = np.delete(Temp_WS, vel_nan_index[:, 0],  0)

# Correct pressure to sea level (ISA)
P_WS = P_WS * (1 - 0.0065 * Z_WS / (Temp_WS + 273.15 + 0.0065 * Z_WS))**(-5.257)

# Certering of location and time fields
x_min = np.min(X_WS)
x_max = np.max(X_WS)
X_WS = X_WS - (x_min + x_max) / 2
y_min = np.min(Y_WS)
y_max = np.max(Y_WS)
Y_WS = Y_WS - (y_min + y_max) / 2
t_min = np.min(T_WS)
t_max = np.max(T_WS)
T_WS = T_WS - t_min # Refer to t = 0

# PINN output grid
T_PINN = T_WS[0 : 1, :] # Same times for reconstruction

# Resolution in degrees
R = 0.2
R_PINN = 6378000 * np.sin(np.radians(R)) # Grid resolution
x_PINN = np.arange(x_min - R_PINN, x_max + R_PINN, R_PINN) # X values in output resolution
y_PINN = np.arange(y_min - R_PINN, y_max + R_PINN, R_PINN) # Y values in output resolution

# Centering of location data
x_PINN = x_PINN - (x_min + x_max) / 2
y_PINN = y_PINN - (y_min + y_max) / 2

# Final output grid
X_PINN, Y_PINN = np.meshgrid(x_PINN, y_PINN)
X_PINN = X_PINN.flatten('F')[:, None]
Y_PINN = Y_PINN.flatten('F')[:, None]

# Dimensions
dim_T_PINN = T_PINN.shape[1]
dim_N_PINN = X_PINN.shape[0]

T_PINN = np.tile(T_PINN, (dim_N_PINN, 1))
X_PINN = np.tile(X_PINN, dim_T_PINN)
Y_PINN = np.tile(Y_PINN, dim_T_PINN)

# Reference values for non-dimensionalization
L = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) # Reference distance
W = np.sqrt(np.nanmax(abs(U_WS)) ** 2 + np.nanmax(abs(V_WS)) ** 2) # Reference velocity
rho = 1.269 # Air density at 15 degrees
nu = 1.382e-5 # Kinematic viscosity at 15 degrees
Re = int(W * L / nu) # Reynolds number
P0 = np.nanmean(P_WS) # Reference pressure level
print('L:', L, 'W', W, 'P0', P0, 'Re', Re)

# Non-dimensionalization
X_WS = X_WS / L
Y_WS = Y_WS / L
T_WS = T_WS * W / L
P_WS = (P_WS - P0) / rho / (W ** 2)
U_WS = U_WS / W
V_WS = V_WS / W

X_PINN = X_PINN / L
Y_PINN = Y_PINN / L
T_PINN = T_PINN * W / L

# Validation cases (remove stations)
# # N_test = 0 # Number of stations to remove
WS_val = np.array([1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 19])
# Choose between different arrays for desired validation case:
# Close: np.array([2, 8, 10, 14, 16, 19])
# Far: np.array([0, 1, 4, 6, 8, 12, 17, 18, 19, 20]) 
# Envelope: np.array([1, 2, 3, 5, 7, 9, 10, 11, 13, 14, 15, 16, 19])

# Remove WS for validation
T_val = T_WS[WS_val, :]
P_val = P_WS[WS_val, :]
U_val = U_WS[WS_val, :]
V_val = V_WS[WS_val, :]
X_val = X_WS[WS_val, :]
Y_val = Y_WS[WS_val, :]
Z_val = Z_WS[WS_val, :]

# Remaining Ws for training
T_WS = np.delete(T_WS, WS_val, 0)
P_WS = np.delete(P_WS, WS_val, 0)
U_WS = np.delete(U_WS, WS_val, 0)
V_WS = np.delete(V_WS, WS_val, 0)
X_WS = np.delete(X_WS, WS_val, 0)
Y_WS = np.delete(Y_WS, WS_val, 0)
print('Number of final weather stations available for training:', T_WS.shape[0])

# Dimensions
dim_N_WS = X_WS.shape[0]
dim_T_WS = X_WS.shape[1]

del WS_data

# Customized dense layer 
class GammaBiasLayer(tf.keras.layers.Layer):
    def __init__(self, units, *args, **kwargs):
        super(GammaBiasLayer, self).__init__(*args, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        
        self.gamma = self.add_weight('gamma',
                                     shape = (self.units,),
                                     initializer = 'ones',
                                     trainable = True)
        
        self.w = tfa.layers.WeightNormalization(Dense(self.units, use_bias = False, 
                                    kernel_initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None),
                                    trainable = True, activation = None))
        

    def call(self, input_tensor):
        return self.gamma * self.w(input_tensor) + self.bias

# Model
num_input_variables = 3 # t, x, y
num_output_variables = 3 # u, v, p

neurons = 200 * num_output_variables 
layers = [num_input_variables] + (2 * (num_input_variables + num_output_variables))*[neurons] + [num_output_variables]

inputs = Input(shape = (num_input_variables, ))
h = GammaBiasLayer(layers[1])(inputs)
h = Activation('tanh')(h) 
for l in layers[2 : 2 * int((len(layers) - 2) / 3)]:
    h = GammaBiasLayer(l)(h)
    h = Activation('tanh')(h)
for l in layers[2 * int((len(layers) - 2) / 3) : -1]:
    h = GammaBiasLayer(layers[-2])(h)
outputs = GammaBiasLayer(layers[-1])(h)

model = Model(inputs = inputs, outputs = outputs)

model.summary()

# Error functions and loss function
mse = tf.keras.losses.MeanSquaredError()
rmse = tf.keras.metrics.RootMeanSquaredError()

@tf.function(reduce_retracing = True)
def loss_NS_2D(model, t_eqns_batch, x_eqns_batch, y_eqns_batch, training):
    mse = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape(persistent = True) as tape1:
        tape1.watch((t_eqns_batch, x_eqns_batch, y_eqns_batch))
        X_eqns_batch = tf.concat([t_eqns_batch, x_eqns_batch, y_eqns_batch], axis = 1)
        Y_eqns_batch = model(X_eqns_batch, training = training) 
        [u_eqns_pred, v_eqns_pred, p_eqns_pred] = tf.split(Y_eqns_batch, num_or_size_splits=Y_eqns_batch.shape[1], axis=1)

    # Derivatives 
    u_t_eqns_pred = tape1.gradient(u_eqns_pred, t_eqns_batch)
    v_t_eqns_pred = tape1.gradient(v_eqns_pred, t_eqns_batch)

    u_x_eqns_pred = tape1.gradient(u_eqns_pred, x_eqns_batch)
    v_x_eqns_pred = tape1.gradient(v_eqns_pred, x_eqns_batch)
    p_x_eqns_pred = tape1.gradient(p_eqns_pred, x_eqns_batch)

    u_y_eqns_pred = tape1.gradient(u_eqns_pred, y_eqns_batch)
    v_y_eqns_pred = tape1.gradient(v_eqns_pred, y_eqns_batch)
    p_y_eqns_pred = tape1.gradient(p_eqns_pred, y_eqns_batch)

    # Navier-Stokes residuals
    e1 = (u_x_eqns_pred + v_y_eqns_pred)
    e2 = (u_t_eqns_pred + (u_eqns_pred * u_x_eqns_pred + v_eqns_pred * u_y_eqns_pred) + p_x_eqns_pred)
    e3 = (v_t_eqns_pred + (u_eqns_pred * v_x_eqns_pred + v_eqns_pred * v_y_eqns_pred) + p_y_eqns_pred)

    return mse(0, e1) + mse(0, e2) + mse(0, e3)

def loss_u(model, t_data_batch, x_data_batch, y_data_batch, u_data_batch, training):
    mse = tf.keras.losses.MeanSquaredError()
    X_data_batch = tf.concat([t_data_batch, x_data_batch, y_data_batch], axis = 1)
    Y_data_batch = model(X_data_batch, training = training) 
    [u_data_pred, _, _] = tf.split(Y_data_batch, num_or_size_splits=Y_data_batch.shape[1], axis=1)

    return mse(u_data_batch, u_data_pred) / tf.math.reduce_std(u_data_batch)**2

def loss_v(model, t_data_batch, x_data_batch, y_data_batch, v_data_batch, training):
    mse = tf.keras.losses.MeanSquaredError()
    X_data_batch = tf.concat([t_data_batch, x_data_batch, y_data_batch], axis = 1)
    Y_data_batch = model(X_data_batch, training = training) 
    [_, v_data_pred, _] = tf.split(Y_data_batch, num_or_size_splits=Y_data_batch.shape[1], axis=1)

    return mse(v_data_batch, v_data_pred) / tf.math.reduce_std(v_data_batch)**2

def loss_p(model, t_data_batch, x_data_batch, y_data_batch, p_data_batch, training):
    mse = tf.keras.losses.MeanSquaredError()
    X_data_batch = tf.concat([t_data_batch, x_data_batch, y_data_batch], axis = 1)
    Y_data_batch = model(X_data_batch, training = training) 
    [_, _, p_data_pred] = tf.split(Y_data_batch, num_or_size_splits=Y_data_batch.shape[1], axis=1)

    return mse(p_data_batch, p_data_pred) / tf.math.reduce_std(p_data_batch)**2 

def loss_total(model, t_u_batch, x_u_batch, y_u_batch, u_u_batch, t_v_batch, x_v_batch, y_v_batch, v_v_batch, t_p_batch, x_p_batch, y_p_batch, p_p_batch, t_eqns_ref_batch, x_eqns_ref_batch, y_eqns_ref_batch, t_eqns_batch, x_eqns_batch, y_eqns_batch, lamb, training):
    NS_eqns = lamb * loss_NS_2D(model, t_eqns_batch, x_eqns_batch, y_eqns_batch, training)
    NS_data = lamb * loss_NS_2D(model, t_eqns_ref_batch, x_eqns_ref_batch, y_eqns_ref_batch, training)
    P_e = loss_p(model, t_p_batch, x_p_batch, y_p_batch, p_p_batch, training)
    U_e = loss_u(model, t_u_batch, x_u_batch, y_u_batch, u_u_batch, training) 
    V_e = loss_v(model, t_v_batch, x_v_batch, y_v_batch, v_v_batch, training) 
    
    total_e = NS_eqns + NS_data + U_e + V_e + P_e

    return  (NS_eqns ** 2 + NS_data**2 + U_e ** 2 + V_e ** 2 + P_e ** 2) / total_e


# Optimize model - gradients:
def grad(model, t_u_batch, x_u_batch, y_u_batch, u_u_batch, t_v_batch, x_v_batch, y_v_batch, v_v_batch,  t_p_batch, x_p_batch, y_p_batch, p_p_batch, t_eqns_ref_batch, x_eqns_ref_batch, y_eqns_ref_batch, t_eqns_batch, x_eqns_batch, y_eqns_batch, lamb):
    with tf.GradientTape() as tape:
        loss_value = loss_total(model, t_u_batch, x_u_batch, y_u_batch, u_u_batch, t_v_batch, x_v_batch, y_v_batch, v_v_batch,  t_p_batch, x_p_batch, y_p_batch, p_p_batch, t_eqns_ref_batch, x_eqns_ref_batch, y_eqns_ref_batch, t_eqns_batch, x_eqns_batch, y_eqns_batch, lamb, training = True)
    gradient_model = tape.gradient(loss_value, model.trainable_variables)
    return loss_value, gradient_model

# Create an optimizer
model_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

# Keep results for plotting
train_loss_results = []
NS_loss_results = []
P_loss_results = []
U_loss_results = []
V_loss_results = []

# Training
num_epochs = 1000 # number of epochs
lamb = 2 # Tuning of physics constraints
batch_PINN = int(np.ceil((dim_N_PINN * dim_T_PINN / n_days * R)))
batch_WS = int(np.ceil(dim_N_WS * dim_T_WS / n_days * R))

# Data dimensions
dim_N_data = dim_N_WS
dim_T_data = dim_T_WS
dim_T_eqns = dim_T_PINN
dim_N_eqns = dim_N_PINN

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_NS_loss_avg = tf.keras.metrics.Mean()
    epoch_P_loss_avg = tf.keras.metrics.Mean()
    epoch_U_loss_avg = tf.keras.metrics.Mean()
    epoch_V_loss_avg = tf.keras.metrics.Mean()

    # Data mixing and shuffling
    idx_t = np.random.choice(dim_T_WS, dim_T_data, replace = False)
    idx_x = np.random.choice(dim_N_WS, dim_N_data, replace = False)
    t_u = T_WS[:, idx_t][idx_x,:].flatten()[:,None]
    x_u = X_WS[:, idx_t][idx_x,:].flatten()[:,None]
    y_u = Y_WS[:, idx_t][idx_x,:].flatten()[:,None]
    z_u = Z_WS[:, idx_t][idx_x,:].flatten()[:,None]
    u_u = U_WS[:, idx_t][idx_x,:].flatten()[:,None]
    v_u = V_WS[:, idx_t][idx_x,:].flatten()[:,None]
    p_u = U_WS[:, idx_t][idx_x,:].flatten()[:,None]

    idx_t = np.random.choice(dim_T_WS, dim_T_data, replace = False)
    idx_x = np.random.choice(dim_N_WS, dim_N_data, replace = False)
    t_v = T_WS[:, idx_t][idx_x,:].flatten()[:,None]
    x_v = X_WS[:, idx_t][idx_x,:].flatten()[:,None]
    y_v = Y_WS[:, idx_t][idx_x,:].flatten()[:,None]
    z_v = Z_WS[:, idx_t][idx_x,:].flatten()[:,None]
    u_v = U_WS[:, idx_t][idx_x,:].flatten()[:,None]
    v_v = V_WS[:, idx_t][idx_x,:].flatten()[:,None]   
    p_v = U_WS[:, idx_t][idx_x,:].flatten()[:,None]

    idx_t = np.random.choice(P_WS.shape[1], P_WS.shape[1], replace = False)
    idx_x = np.random.choice(P_WS.shape[0], P_WS.shape[0], replace = False)
    t_p = T_WS[:, idx_t][idx_x,:].flatten()[:,None]
    x_p = X_WS[:, idx_t][idx_x,:].flatten()[:,None]
    y_p = Y_WS[:, idx_t][idx_x,:].flatten()[:,None]
    z_p = Z_WS[:, idx_t][idx_x,:].flatten()[:,None]
    u_p = U_WS[:, idx_t][idx_x,:].flatten()[:,None]
    v_p = V_WS[:, idx_t][idx_x,:].flatten()[:,None]
    p_p = P_WS[:, idx_t][idx_x,:].flatten()[:,None]

    idx_t = np.random.choice(dim_T_PINN, dim_T_eqns, replace = False)
    idx_x = np.random.choice(dim_N_PINN, dim_N_eqns, replace = False)
    t_eqns = T_PINN[:, idx_t][idx_x,:].flatten()[:,None]
    x_eqns = X_PINN[:, idx_t][idx_x,:].flatten()[:,None]
    y_eqns = Y_PINN[:, idx_t][idx_x,:].flatten()[:,None]

    idx_t = np.random.choice(dim_T_WS, dim_T_data, replace = False)
    idx_x = np.random.choice(dim_N_WS, dim_N_data, replace = False)
    t_eqns_ref = T_WS[:, idx_t][idx_x,:].flatten()[:,None]
    x_eqns_ref = X_WS[:, idx_t][idx_x,:].flatten()[:,None]
    y_eqns_ref = Y_WS[:, idx_t][idx_x,:].flatten()[:,None]

    idx_batch = np.random.choice(t_u.shape[0], t_u.shape[0], replace = False)
    t_u = t_u[idx_batch, :]
    x_u = x_u[idx_batch, :]
    y_u = y_u[idx_batch, :]
    u_u = u_u[idx_batch, :]
    idx_batch = np.random.choice(t_v.shape[0], t_v.shape[0], replace = False)
    t_v = t_v[idx_batch, :]
    x_v = x_v[idx_batch, :]
    y_v = y_v[idx_batch, :]
    v_v = v_v[idx_batch, :]
    idx_batch = np.random.choice(t_p.shape[0], t_p.shape[0], replace = False)
    t_p = t_p[idx_batch, :]
    x_p = x_p[idx_batch, :]
    y_p = y_p[idx_batch, :]
    p_p = p_p[idx_batch, :]
    idx_batch = np.random.choice(t_eqns.shape[0], t_eqns.shape[0], replace = False)
    t_eqns = t_eqns[idx_batch, :]
    x_eqns = x_eqns[idx_batch, :]
    y_eqns = y_eqns[idx_batch, :]
    idx_batch = np.random.choice(t_eqns_ref.shape[0], t_eqns_ref.shape[0], replace = False)
    t_eqns_ref = t_eqns_ref[idx_batch, :]
    x_eqns_ref = x_eqns_ref[idx_batch, :]
    y_eqns_ref = y_eqns_ref[idx_batch, :]

    # Remove remaining NaN
    nan_index = np.argwhere(np.isnan(u_u))
    t_u = np.delete(t_u, nan_index[:, 0],  0)
    u_u = np.delete(u_u, nan_index[:, 0],  0)
    x_u = np.delete(x_u, nan_index[:, 0],  0)
    y_u = np.delete(y_u, nan_index[:, 0],  0)
    nan_index = np.argwhere(np.isnan(v_v))
    t_v = np.delete(t_v, nan_index[:, 0],  0)
    v_v = np.delete(v_v, nan_index[:, 0],  0)
    x_v = np.delete(x_v, nan_index[:, 0],  0)
    y_v = np.delete(y_v, nan_index[:, 0],  0)
    nan_index = np.argwhere(np.isnan(p_p))
    t_p = np.delete(t_p, nan_index[:, 0],  0)
    p_p = np.delete(p_p, nan_index[:, 0],  0)
    x_p = np.delete(x_p, nan_index[:, 0],  0)
    y_p = np.delete(y_p, nan_index[:, 0],  0)

    # Batch size distribution
    div_u = range(0, len(x_u), batch_WS)
    div_v = range(0, len(x_v), batch_WS)
    div_p = range(0, len(x_p), batch_WS)
    div_eqns = range(0, len(x_eqns_ref), batch_WS)
    div_PINN = range(0, len(x_eqns), batch_PINN)

    min_div = min([len(div_u), len(div_v), len(div_p), len(div_eqns), len(div_PINN)])

    # Batch step
    for index in range(0, min_div):
        index_u = div_u[index]
        index_v = div_v[index]
        index_p = div_p[index]
        index_eqns = div_eqns[index]
        index_PINN = div_PINN[index]
        t_u_batch = tf.convert_to_tensor(t_u[index_u : index_u + batch_WS, :], dtype = 'float32')
        x_u_batch = tf.convert_to_tensor(x_u[index_u : index_u + batch_WS, :], dtype = 'float32')
        y_u_batch = tf.convert_to_tensor(y_u[index_u : index_u + batch_WS, :], dtype = 'float32')
        u_u_batch = tf.convert_to_tensor(u_u[index_u : index_u + batch_WS, :], dtype = 'float32')
        v_u_batch = tf.convert_to_tensor(v_u[index_u : index_u + batch_WS, :], dtype = 'float32')
        t_v_batch = tf.convert_to_tensor(t_v[index_v : index_v + batch_WS, :], dtype = 'float32')
        x_v_batch = tf.convert_to_tensor(x_v[index_v : index_v + batch_WS, :], dtype = 'float32')
        y_v_batch = tf.convert_to_tensor(y_v[index_v : index_v + batch_WS, :], dtype = 'float32')
        u_v_batch = tf.convert_to_tensor(u_v[index_v : index_v + batch_WS, :], dtype = 'float32')
        v_v_batch = tf.convert_to_tensor(v_v[index_v : index_v + batch_WS, :], dtype = 'float32')
        t_p_batch = tf.convert_to_tensor(t_p[index_p : index_p + batch_WS, :], dtype = 'float32')
        x_p_batch = tf.convert_to_tensor(x_p[index_p : index_p + batch_WS, :], dtype = 'float32')
        y_p_batch = tf.convert_to_tensor(y_p[index_p : index_p + batch_WS, :], dtype = 'float32')
        u_p_batch = tf.convert_to_tensor(u_p[index_p : index_p + batch_WS, :], dtype = 'float32')
        v_p_batch = tf.convert_to_tensor(v_p[index_p : index_p + batch_WS, :], dtype = 'float32')
        p_p_batch = tf.convert_to_tensor(p_p[index_p : index_p + batch_WS, :], dtype = 'float32')
        t_eqns_ref_batch = tf.convert_to_tensor(t_eqns_ref[index_eqns : index_eqns + batch_WS, :], dtype = 'float32')
        x_eqns_ref_batch = tf.convert_to_tensor(x_eqns_ref[index_eqns : index_eqns + batch_WS, :], dtype = 'float32')
        y_eqns_ref_batch = tf.convert_to_tensor(y_eqns_ref[index_eqns : index_eqns + batch_WS, :], dtype = 'float32')
        t_eqns_batch = tf.convert_to_tensor(t_eqns[index_PINN : index_PINN + batch_PINN, :], dtype = 'float32')
        x_eqns_batch = tf.convert_to_tensor(x_eqns[index_PINN : index_PINN + batch_PINN, :], dtype = 'float32')
        y_eqns_batch = tf.convert_to_tensor(y_eqns[index_PINN : index_PINN + batch_PINN, :], dtype = 'float32')
        
        loss_train, grads = grad(model, t_u_batch, x_u_batch, y_u_batch, u_u_batch, t_v_batch, x_v_batch, y_v_batch, v_v_batch,  t_p_batch, x_p_batch, y_p_batch, p_p_batch, t_eqns_ref_batch, x_eqns_ref_batch, y_eqns_ref_batch, t_eqns_batch, x_eqns_batch, y_eqns_batch, lamb)

        NS_loss = loss_NS_2D(model, t_eqns_batch, x_eqns_batch, y_eqns_batch, training = False)
        P_loss = loss_p(model, t_p_batch, x_p_batch, y_p_batch, p_p_batch, training = False)
        U_loss = loss_u(model, t_u_batch, x_u_batch, y_u_batch, u_u_batch, training = False) 
        V_loss = loss_v(model, t_v_batch, x_v_batch, y_v_batch, v_v_batch, training = False) 
    
        model_optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg.update_state(loss_train)
        epoch_NS_loss_avg.update_state(NS_loss)
        epoch_P_loss_avg.update_state(P_loss)
        epoch_U_loss_avg.update_state(U_loss)
        epoch_V_loss_avg.update_state(V_loss)
     
    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    NS_loss_results.append(epoch_NS_loss_avg.result())
    P_loss_results.append(epoch_P_loss_avg.result())
    U_loss_results.append(epoch_U_loss_avg.result())
    V_loss_results.append(epoch_V_loss_avg.result())

    # Update learning rate (adaptive)
    if epoch_loss_avg.result() > 1e-1:
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    elif epoch_loss_avg.result() > 3e-2:
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    elif epoch_loss_avg.result() > 3e-3:
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    else:
        model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)

    print("Epoch: {:d} Loss_training: {:.3e} NS_loss: {:.3e} P_loss: {:.3e} U_loss: {:.3e} V_loss: {:.3e}".format(epoch, epoch_loss_avg.result(), 
    epoch_NS_loss_avg.result(), epoch_P_loss_avg.result(), epoch_U_loss_avg.result(), epoch_V_loss_avg.result()))
    
    ################# Save Data ###########################
    if (epoch + 1) % num_epochs == 0:
        # Output in higher resolution
        U_PINN = np.zeros_like(X_PINN)
        V_PINN = np.zeros_like(X_PINN)
        P_PINN = np.zeros_like(X_PINN)
        # Values predicted on WS locations
        U_WS_pred = np.zeros_like(X_WS)
        V_WS_pred = np.zeros_like(X_WS)
        P_WS_pred = np.zeros_like(X_WS)
        # Values predicted on validation set
        U_val_pred = np.zeros_like(X_val)
        V_val_pred = np.zeros_like(X_val)
        P_val_pred = np.zeros_like(X_val)

        for snap in range(0, dim_T_PINN):
            t_out = T_PINN[:, snap : snap + 1]
            x_out = X_PINN[:, snap : snap + 1]
            y_out = Y_PINN[:, snap : snap + 1]

            X_out = tf.concat([t_out, x_out, y_out], 1)

            # Prediction
            Y_out = model(X_out, training = False)
            [u_pred_out, v_pred_out, p_pred_out] = tf.split(Y_out, num_or_size_splits = Y_out.shape[1], axis=1)

            U_PINN[:,snap : snap + 1] = u_pred_out
            V_PINN[:,snap : snap + 1] = v_pred_out
            P_PINN[:,snap : snap + 1] = p_pred_out

        for snap in range(0, dim_T_WS):
            t_out = T_WS[:, snap : snap + 1]
            x_out = X_WS[:, snap : snap + 1]
            y_out = Y_WS[:, snap : snap + 1]

            X_out = tf.concat([t_out, x_out, y_out], 1)

            # Prediction
            Y_out = model(X_out, training = False)
            [u_pred_out, v_pred_out, p_pred_out] = tf.split(Y_out, num_or_size_splits = Y_out.shape[1], axis=1)

            U_WS_pred[:,snap : snap + 1] = u_pred_out
            V_WS_pred[:,snap : snap + 1] = v_pred_out
            P_WS_pred[:,snap : snap + 1] = p_pred_out

        for snap in range(0, T_val.shape[1]):
            t_out = T_val[:, snap : snap + 1]
            x_out = X_val[:, snap : snap + 1]
            y_out = Y_val[:, snap : snap + 1]

            X_out = tf.concat([t_out, x_out, y_out], 1)

            # Prediction
            Y_out = model(X_out, training = False)
            [u_pred_out, v_pred_out, p_pred_out] = tf.split(Y_out, num_or_size_splits = Y_out.shape[1], axis=1)

            U_val_pred[:,snap : snap + 1] = u_pred_out
            V_val_pred[:,snap : snap + 1] = v_pred_out
            P_val_pred[:,snap : snap + 1] = p_pred_out

        # Save data in .mat file (dimensionless units)
        scipy.io.savemat('Brussels_%s_lambda_%s_R_%s_envelope.mat' %(str(epoch + 1), str(lamb), str(R)),
                            {'T_PINN': T_PINN, 'X_PINN': X_PINN, 'Y_PINN': Y_PINN, 'U_PINN': U_PINN, 'V_PINN': V_PINN, 'P_PINN': P_PINN,
                            'T_WS': T_WS, 'X_WS': X_WS, 'Y_WS': Y_WS, 'U_WS': U_WS, 'V_WS': V_WS, 'P_WS': P_WS,
                            'U_WS_pred': U_WS_pred, 'V_WS_pred': V_WS_pred, 'P_WS_pred': P_WS_pred,
                            'T_val': T_val, 'X_val': X_val, 'Y_val': Y_val, 'U_val': U_val, 'V_val': V_val, 'P_val': P_val,
                            'U_val_pred': U_val_pred, 'V_val_pred': V_val_pred, 'P_valt_pred': P_val_pred,
                            'Train_loss' : train_loss_results, 'NS_loss' : NS_loss_results,
                            'P_loss' : P_loss_results, 'U_loss' : U_loss_results, 'V_loss' : V_loss_results}) # Change ending of the .mat name according to validation case selected: close, far or envelope

print('Process completed')