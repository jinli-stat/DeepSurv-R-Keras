library(tensorflow)
library(keras)
library(survival)
# BiocManager::install("survcomp")

neg_log_likelihd <- function(y_true, y_pred) {
  event <- y_true[, 1]
  time <- y_true[, 2]
  mask <- k_cast(time <= k_reshape(time, shape = c(-1, 1)), dtype = "float32")
  
  log_loss <- k_log(k_sum(mask * k_exp(y_pred), axis = 1))
  neg_log_loss <- -k_sum(event * (y_pred - log_loss))
  
  return(neg_log_loss / k_sum(event))
}


build_deepsurv <- function(num_input, 
                           num_layer, 
                           num_nodes, 
                           string_activation, 
                           num_l2, 
                           num_dropout, 
                           num_lr, 
                           lr_decay) {
  
  input_layer <- layer_input(shape = c(num_input),
                             dtype = "float32")
  
  hidden_layers <- input_layer
  
  for (i in 1:num_layer) {
    hidden_layers <- hidden_layers %>%
      layer_dense(units = num_nodes, 
                  activation = string_activation, 
                  # kernel_regularizer = regularizer_l2(num_l2),
                  kernel_initializer = "glorot_uniform") %>%
      layer_dropout(rate = num_dropout)
  }
  
  hidden_layers <- hidden_layers %>%
    layer_dense(units = 1, 
                activation = "linear", 
                kernel_regularizer = regularizer_l2(num_l2)) %>% 
    layer_activity_regularization(l2 = num_l2)
  
  model <- keras_model(input_layer, hidden_layers)
  
  model %>% compile(
    loss = neg_log_likelihd,
    optimizer = optimizer_nadam(learning_rate = num_lr, 
                                decay = lr_decay, 
                                clipnorm=1.)
  )
}

c_index <- function(LP_pred, Y, E){
  c_index <- survcomp::concordance.index(LP_pred, Y, E)$c.index
  return(c_index)
}
