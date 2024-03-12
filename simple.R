library(reticulate)
library(tensorflow)
library(keras)
library(magrittr)
library(survival)
library(ggplot2)

# BiocManager::install("rhdf5")
X_train <- rhdf5::h5read("Data/whas.h5", "train/x") %>% t()
event_train <- rhdf5::h5read("Data/whas.h5", "train/e") %>% c()
time_train <- rhdf5::h5read("Data/whas.h5", "train/t") %>% c()
Y_train <- cbind(event_train, time_train)

X_test <- rhdf5::h5read("Data/whas.h5", "test/x") %>% t()
event_test <- rhdf5::h5read("Data/whas.h5", "test/e") %>% c()
time_test <- rhdf5::h5read("Data/whas.h5", "test/t") %>% c()
Y_test <- cbind(event_test, time_test)

n_patients_train <- dim(X_train)[1]
n_features <- dim(X_train)[2]

m1 <- colMeans(X_train)
s1 <- apply(X_train, 2, sd)
X_train <- scale(X_train, center = m1, scale = s1)
X_test <- scale(X_test, center = m1, scale = s1)

neg_log_likelihd <- function(y_true, y_pred) {
  event <- y_true[, 1]
  time <- y_true[, 2]
  mask <- k_cast(time <= k_reshape(time, shape = c(-1, 1)), dtype = "float32")

  log_loss <- k_log(k_sum(mask * k_exp(y_pred), axis = 1))
  neg_log_loss <- -k_sum(event * (y_pred - log_loss))
  return(neg_log_loss / k_sum(event))
}

activation <- "relu"
n_nodes <- 48
learning_rate <- 0.067
l2_reg <- 16.094
dropout <- 0.147
lr_decay <- 6.494e-4
momentum <- 0.863

model <- keras_model_sequential()
model %>%
  layer_dense(units = n_features, activation = activation, kernel_initializer = "glorot_uniform", input_shape = c(n_features)) %>%
  layer_dropout(dropout) %>%
  layer_dense(units = n_nodes, activation = activation, kernel_initializer = "glorot_uniform") %>%
  layer_dropout(dropout) %>%
  layer_dense(units = 1, activation = "linear", kernel_initializer = "glorot_uniform", kernel_regularizer = regularizer_l2(l2_reg)) %>%
  layer_activity_regularization(l2 = l2_reg)

# optimizer_nadam
optimizer <- optimizer_nadam(learning_rate = learning_rate, decay = lr_decay, clipnorm = 1.)

# Compile the model and show a summary of it
model %>% compile(loss = neg_log_likelihd, optimizer = optimizer)
# negative_log_likelihood(E_train)

es <- callback_early_stopping(monitor = "loss", patience = 20, verbose = 1)

# Add the callback to a list
callbacks_list <- list(es)

epochs <- 500
history <- model %>% fit(X_train, Y_train,
  batch_size = 1000,
  epochs = epochs,
  shuffle = TRUE,
  callbacks = callbacks_list
)

LP_train <- model %>% predict(X_train)
LP_test <- model %>% predict(X_test)

# BiocManager::install("survcomp")
c_index_train <- survcomp::concordance.index(LP_train, Y_train[, 2], Y_train[, 1])$c.index
cat(paste0("c-index of training dataset = ", c_index_train, "\n"))

c_index_test <- survcomp::concordance.index(model %>% predict(X_test), Y_test[, 2], Y_test[, 1])$c.index
cat(paste0("c-index of testing dataset = ", c_index_test, "\n"))

# Y_pred_train <- exp(-(model %>% predict(X_train)))
# lifelines <- import("lifelines")
# c_index_train <- lifelines$utils$concordance_index(Y_train, Y_pred_train, E_train)
# cat(paste0("c-index of training dataset = ", c_index_train, "\n"))

# install.packages("mlr3proba", repos = "https://mlr-org.r-universe.dev")
# library(mlr3proba)
# blhz <- breslow(Y_train, E_train, LP_train, LP_test[3], eval_times = 2, type = "cumhaz")
