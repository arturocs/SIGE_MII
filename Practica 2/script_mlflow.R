

library(keras)
library(mlflow)

hidden_units      <- mlflow_param("hidden_units", 100, "integer", "Number of units of the hidden layer")
hidden_activation <- mlflow_param("hidden_activation", "sigmoid", "string", "Activation function for the hidden layer")
dropout           <- mlflow_param("dropout", 0.3, "numeric", "Dropout rate (after the hidden layer)")
epsilon           <- mlflow_param("epsilon", 0.01, "numeric", "Epsilon parameter of the batch normalization (after convolution)")
batch_size        <- mlflow_param("batch_size", 128, "integer", "Mini-batch size")
epochs            <- mlflow_param("epochs", 5, "integer", "Number of training epochs")

## -------------------------------------------------------------------------------------
## Cargar y pre-procesar datos

dataset_dir           <- './data/images/medium10000_twoClasses/'
train_images_dir      <- paste0(dataset_dir, 'train')
val_images_dir        <- paste0(dataset_dir, 'val')
test_images_dir       <- paste0(dataset_dir, 'test')
train_images_generator <- image_data_generator(rescale = 1/255)
val_images_generator   <- image_data_generator(rescale = 1/255)
test_images_generator  <- image_data_generator(rescale = 1/255)

train_generator_flow <- flow_images_from_directory(
  directory = train_images_dir,
  generator = train_images_generator,
  class_mode = 'categorical',
  batch_size = 128,
  target_size = c(64, 64)         # (w x h) --> (64 x 64)
)

validation_generator_flow <- flow_images_from_directory(
  directory = val_images_dir,
  generator = val_images_generator,
  class_mode = 'categorical',
  batch_size = 128,
  target_size = c(64, 64)         # (w x h) --> (64 x 64)
)

test_generator_flow <- flow_images_from_directory(
  directory = test_images_dir,
  generator = test_images_generator,
  class_mode = 'categorical',
  batch_size = 128,
  target_size = c(64, 64)         # (w x h) --> (64 x 64)
)


## -------------------------------------------------------------------------------------
## Crear modelo

# Definir arquitectura
model <- keras_model_sequential() 
model %>% 
 layer_conv_2d(filters = 20, kernel_size = c(5, 5), activation = "relu", input_shape = c(28, 28, 1)) %>%
 layer_batch_normalization(epsilon = epsilon) %>%
 layer_max_pooling_2d(pool_size = c(2, 2)) %>%
 layer_flatten() %>%
 layer_dense(units = hidden_units, activation = hidden_activation) %>%
 layer_dropout(rate = dropout) %>%
 layer_dense(units = 10, activation = "softmax")

summary(model)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu", input_shape = c(64, 64, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dense(units = 2, activation = "softmax")

summary(model)

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(
    lr = 0.0001,
    beta_1 = 0.9,
    beta_2 = 0.999,
    epsilon = NULL,
    decay = 0,
    amsgrad = FALSE,
    clipnorm = NULL,
    clipvalue = NULL
  ),
  metrics = c('accuracy'))

## -------------------------------------------------------------------------------------
## MLflow
with(mlflow_start_run(), {

  # Entrenar modelo
  
  history <- model %>% 
    fit_generator(
      generator = train_generator_flow, 
      validation_data = validation_generator_flow,
      steps_per_epoch = 10,
      validation_steps = 1,
      epochs = 10
    )

  plot(history)
  
  # Calcular metricas sobre datos de validación

  metrics <- model %>% 
    evaluate_generator(test_generator_flow, steps = 1)
  
  # Guardar valores interesantes de la ejecución
  # Por ejemplo, para estudio de dropout + epochs
  mlflow_log_param("dropout", dropout)
  mlflow_log_param("epochs", epochs)
  mlflow_log_metric("loss", metrics["loss"])
  mlflow_log_metric("accuracy", metrics["accuracy"])
  
  # Guardar modelo
  mlflow_log_model(model, "model")
  
  # Mostrar salida
  message("CNN model (dropout=", dropout, ", epochs=", epochs, "):")
  message("  loss: ", metrics["loss"])
  message("  accuracy: ", metrics["accuracy"])
})