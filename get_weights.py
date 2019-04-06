import h5py
from keras.models import load_model

def print_keras_wegiths(weight_file_path):
    f = h5py.File(weight_file_path)  
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))  

        for layer, g in f.items():  
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items(): 
                print("      {}: {}".format(key, value))  

            print("    Dataset:")
            for name, d in g.items():
                print("      {}: {}".format(name, d.value.shape)) 
                print("      {}: {}".format(name. d.value))
    finally:
        f.close()




model = load_model("live_model.h5")
#model1 ={"class_name": "Sequential", "config": {"layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_1", "bias_constraint": "null", "filters": 16, "dtype": "float32", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"seed": "null", "mode": "fan_avg", "scale": 1.0, "distribution": "uniform"}}, "use_bias": "true", "batch_input_shape": ["null", 32, 32, 3], "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": "null", "kernel_size": [3, 3], "kernel_constraint": "null", "trainable": "true", "activity_regularizer": "null", "data_format": "channels_last", "strides": [1, 1], "activation": "linear", "bias_regularizer": "null", "dilation_rate": [1, 1]}}, {"class_name": "Activation", "config": {"trainable": "true", "activation": "relu", "name": "activation_1"}}, {"class_name": "BatchNormalization", "config": {"gamma_regularizer": "null", "beta_regularizer": "null", "scale": "true", "momentum": 0.99, "center": "true", "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_constraint": "null", "epsilon": 0.001, "trainable": "true", "beta_constraint": "null", "axis": -1, "name": "batch_normalization_1", "moving_mean_initializer": {"class_name": "Zeros", "config": {}}}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "bias_constraint": "null", "filters": 16, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"seed": "null", "mode": "fan_avg", "scale": 1.0, "distribution": "uniform"}}, "use_bias": "true", "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": "null", "kernel_size": [3, 3], "kernel_constraint": "null", "trainable": "true", "activity_regularizer": "null", "data_format": "channels_last", "strides": [1, 1], "activation": "linear", "bias_regularizer": "null", "dilation_rate": [1, 1]}}, {"class_name": "Activation", "config": {"trainable": "true", "activation": "relu", "name": "activation_2"}}, {"class_name": "BatchNormalization", "config": {"gamma_regularizer": "null", "beta_regularizer": "null", "scale": "true", "momentum": 0.99, "center": "true", "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_constraint": "null", "epsilon": 0.001, "trainable": "true", "beta_constraint": "null", "axis": -1, "name": "batch_normalization_2", "moving_mean_initializer": {"class_name": "Zeros", "config": {}}}}, {"class_name": "MaxPooling2D", "config": {"pool_size": [2, 2], "data_format": "channels_last", "trainable": "true", "strides": [2, 2], "padding": "valid", "name": "max_pooling2d_1"}}, {"class_name": "Dropout", "config": {"seed": "null", "rate": 0.25, "trainable": "true", "name": "dropout_1", "noise_shape": "null"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "bias_constraint": "null", "filters": 32, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"seed": "null", "mode": "fan_avg", "scale": 1.0, "distribution": "uniform"}}, "use_bias": "true", "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": "null", "kernel_size": [3, 3], "kernel_constraint": "null", "trainable": "true", "activity_regularizer": "null", "data_format": "channels_last", "strides": [1, 1], "activation": "linear", "bias_regularizer": "null", "dilation_rate": [1, 1]}}, {"class_name": "Activation", "config": {"trainable": "true", "activation": "relu", "name": "activation_3"}}, {"class_name": "BatchNormalization", "config": {"gamma_regularizer": "null", "beta_regularizer": "null", "scale": "true", "momentum": 0.99, "center": "true", "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_constraint": "null", "epsilon": 0.001, "trainable": "true", "beta_constraint": "null", "axis": -1, "name": "batch_normalization_3", "moving_mean_initializer": {"class_name": "Zeros", "config": {}}}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "bias_constraint": "null", "filters": 32, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"seed": "null", "mode": "fan_avg", "scale": 1.0, "distribution": "uniform"}}, "use_bias": "true", "padding": "same", "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": "null", "kernel_size": [3, 3], "kernel_constraint": "null", "trainable": "true", "activity_regularizer": "null", "data_format": "channels_last", "strides": [1, 1], "activation": "linear", "bias_regularizer": "null", "dilation_rate": [1, 1]}}, {"class_name": "Activation", "config": {"trainable": "true", "activation": "relu", "name": "activation_4"}}, {"class_name": "BatchNormalization", "config": {"gamma_regularizer": "null", "beta_regularizer": "null", "scale": "true", "momentum": 0.99, "center": "true", "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_constraint": "null", "epsilon": 0.001, "trainable": "true", "beta_constraint": "null", "axis": -1, "name": "batch_normalization_4", "moving_mean_initializer": {"class_name": "Zeros", "config": {}}}}, {"class_name": "MaxPooling2D", "config": {"pool_size": [2, 2], "data_format": "channels_last", "trainable": "true", "strides": [2, 2], "padding": "valid", "name": "max_pooling2d_2"}}, {"class_name": "Dropout", "config": {"seed": "null", "rate": 0.25, "trainable": "true", "name": "dropout_2", "noise_shape": "null"}}, {"class_name": "Flatten", "config": {"data_format": "channels_last", "name": "flatten_1", "trainable": "true"}}, {"class_name": "Dense", "config": {"bias_constraint": "null", "activity_regularizer": "null", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"seed": "null", "mode": "fan_avg", "scale": 1.0, "distribution": "uniform"}}, "units": 64, "bias_initializer": {"class_name": "Zeros", "config": {}}, "activation": "linear", "kernel_constraint": "null", "trainable": "true", "use_bias": "true", "kernel_regularizer": "null", "name": "dense_1", "bias_regularizer": "null"}}, {"class_name": "Activation", "config": {"trainable": "true", "activation": "relu", "name": "activation_5"}}, {"class_name": "BatchNormalization", "config": {"gamma_regularizer": "null", "beta_regularizer": "null", "scale": "true", "momentum": 0.99, "center": "true", "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_constraint": "null", "epsilon": 0.001, "trainable": "true", "beta_constraint": "null", "axis": -1, "name": "batch_normalization_5", "moving_mean_initializer": {"class_name": "Zeros", "config": {}}}}, {"class_name": "Dropout", "config": {"seed": "null", "rate": 0.5, "trainable": "true", "name": "dropout_3", "noise_shape": "null"}}, {"class_name": "Dense", "config": {"bias_constraint": "null", "activity_regularizer": "null", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"seed": "null", "mode": "fan_avg", "scale": 1.0, "distribution": "uniform"}}, "units": 2, "bias_initializer": {"class_name": "Zeros", "config": {}}, "activation": "linear", "kernel_constraint": "null", "trainable": "true", "use_bias": "true", "kernel_regularizer": "null", "name": "dense_2", "bias_regularizer": "null"}}, {"class_name": "Activation", "config": {"trainable": "true", "activation": "softmax", "name": "activation_6"}}], "name": "sequential_1"}}

#This script shows the model/architecture of my CNN
#Type model.get_weights() to see weights of my CNN
print(model.get_weights)

