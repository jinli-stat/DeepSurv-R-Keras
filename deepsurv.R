library(tensorflow)
library(keras)
library(survival)
# BiocManager::install("survcomp")

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

build_deepsurv <- function(num_input, num_layer, num_nodes, string_activation, num_l2, num_dropout, num_lr, lr_decay) {
  input_layer <- layer_input(shape = c(num_input),
                             dtype = "float32")
  
  hidden_layers <- input_layer
  
  for (i in 1:num_layer) {
    hidden_layers <- hidden_layers %>%
      layer_dense(units = num_nodes, activation = string_activation, kernel_regularizer = regularizer_l2(num_l2)) %>%
      layer_dropout(rate = num_dropout)
  }
  
  hidden_layers <- hidden_layers %>%
    layer_dense(units = 1, activation = NULL)
  
  model <- keras_model(input_layer, hidden_layers)
  
  model %>% compile(
    optimizer = optimizer_nadam(learning_rate = num_lr, decay = lr_decay, clipnorm=1.), 
    loss = negative_log_likelihood(E_train),
    metrics = NULL
  )

}


model = build_deepsurv(num_input=6,
                  num_layer= 3,
                  string_activation ="relu",
                  num_nodes = 48,
                  num_lr = 0.067,
                  num_l2 = 16.094,
                  num_dropout = 0.147,
                  lr_decay = 6.494e-4)

es <- callback_early_stopping(monitor = "loss", patience = 5, verbose = 1)

# Add the callback to a list
callbacks_list <- list(es)

epochs <- 30
history <- model %>% fit(X_train, Y_train,
                         batch_size = n_patients_train,
                         epochs = epochs,
                         shuffle = FALSE,
                         callbacks = callbacks_list
)

LP_train <- model %>% predict(X_train)
LP_test <- model %>% predict(X_test)


c_index <- function(X, Y, E){
  LP_pred <- model %>% predict(X)
  c_ind <- survcomp::concordance.index(LP_pred, Y, E)$c.index
  return(c_ind)
}
c_index(X_train, Y_train, E_train)
c_index(X_test, Y_test, E_test)
