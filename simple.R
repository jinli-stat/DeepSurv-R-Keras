library(reticulate)
library(tensorflow)
library(keras)
library(magrittr)
library(survival)
library(ggplot2)

# BiocManager::install("rhdf5")
X_train <- rhdf5::h5read("Data/whas.h5", "train/x") %>% t()
E_train <- rhdf5::h5read("Data/whas.h5", "train/e") %>% c()
Y_train <- rhdf5::h5read("Data/whas.h5", "train/t") %>% c()

X_test <- rhdf5::h5read("Data/whas.h5", "test/x") %>% t()
E_test <- rhdf5::h5read("Data/whas.h5", "test/e") %>% c()
Y_test <- rhdf5::h5read("Data/whas.h5", "test/t") %>% c()


n_patients_train <- dim(X_train)[1]
n_features <- dim(X_train)[2]

m1 <- colMeans(X_train)
s1 <- apply(X_train, 2, sd)
X_train <- scale(X_train, center = m1, scale = s1)
X_test <- scale(X_test, center = m1, scale = s1)

sort_idx <- order(Y_train, decreasing = TRUE)
X_train <- X_train[sort_idx, ]
Y_train <- Y_train[sort_idx]
E_train <- E_train[sort_idx]

negative_log_likelihood <- function(E) {
  function(y_true, y_pred) {
    hazard_ratio <- k_exp(y_pred)
    log_risk <- k_log(k_cumsum(hazard_ratio))
    uncensored_likelihood <- k_transpose(y_pred) - log_risk
    censored_likelihood <- uncensored_likelihood * E
    neg_likelihood_ <- -k_sum(censored_likelihood)
    
    # num_observed_events <- k_cumsum(E)
    # num_observed_events <- k_cast(num_observed_events, dtype = "float32")

    num_observed_events <- k_constant(1, dtype = "float32")

    neg_likelihood <- neg_likelihood_ / num_observed_events

    return(neg_likelihood)
  }
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
optimizer <- optimizer_nadam(learning_rate = learning_rate, decay = lr_decay, clipnorm=1.)

# Compile the model and show a summary of it
model %>% compile(loss = negative_log_likelihood(E_train), optimizer = optimizer)


es <- callback_early_stopping(monitor = "loss", patience = 20, verbose = 1)

# Add the callback to a list
callbacks_list <- list(es)

epochs <- 500
history <- model %>% fit(X_train, Y_train,
  batch_size = n_patients_train,
  epochs = epochs,
  shuffle = FALSE,
  callbacks = callbacks_list
)

# Y_pred_train <- exp(-(model %>% predict(X_train)))
# lifelines <- import("lifelines")
# c_index_train <- lifelines$utils$concordance_index(Y_train, Y_pred_train, E_train)
# cat(paste0("c-index of training dataset = ", c_index_train, "\n"))


LP_train <- model %>% predict(X_train)
LP_test <- model %>% predict(X_test)

# BiocManager::install("survcomp")
c_index_train <- survcomp::concordance.index(LP_train, Y_train, E_train)$c.index
cat(paste0("c-index of training dataset = ", c_index_train, "\n"))

c_index_test <- survcomp::concordance.index(model %>% predict(X_test), Y_test, E_test)$c.index
cat(paste0("c-index of testing dataset = ", c_index_test, "\n"))


# install.packages("mlr3proba", repos = "https://mlr-org.r-universe.dev")
library(mlr3proba)
blhz <- breslow(Y_train, E_train, LP_train, LP_test[3], eval_times=2, type = "cumhaz")

