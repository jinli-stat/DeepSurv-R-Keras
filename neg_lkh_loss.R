library(tensorflow)
library(keras)

make_riskset <- function(time) {
  
  n <- length(time)
  
  indices <- order(time, decreasing = TRUE)
  
  risk_set <- matrix(FALSE, nrow = n, ncol = n)
  
  for (i_org in seq_along(indices)) {
    i_sort <- indices[i_org]
    ti <- time[i_sort]
    k <- i_org
    while (k <= n && ti == time[indices[k]]) {
      k <- k + 1
    }
    risk_set[i_sort, indices[1:(k-1)]] <- TRUE
  }
  return(risk_set)
}

logsumexp_masked <- function(risk_scores, mask, axis = 1, keepdims = NULL) {
  mask_f <- tf$cast(mask, dtype = tf$float32)
  
  risk_scores <- t(risk_scores)
  
  risk_scores_masked <- risk_scores * mask_f
  
  amax <- tf$reduce_max(risk_scores_masked, axis = axis, keepdims = TRUE)
  
  risk_scores_shift <- risk_scores_masked - amax
  
  exp_masked <- tf$exp(risk_scores_shift) * mask_f
  
  exp_sum <- tf$reduce_sum(exp_masked, axis = axis, keepdims = keepdims)
  
  output <- amax + tf$log(exp_sum)
  return(output)
}

coxph_loss_ <- function(event, riskset, predictions) {
  rr <- logsumexp_masked(predictions, riskset, axis = 1)
  
  event <- tf$cast(event, dtype = rr$dtype)
  
  losses <- event * (rr - predictions)
  
  loss <- tf$reduce_mean(losses)
  return(loss)
}

coxph_loss <- function(predictions, outcome) {
  time <- outcome[, 2]
  event <- tf$expand_dims(outcome[, 1], axis = 1)
  
  risk_set <- make_riskset(time)
  
  return(coxph_loss_(event, risk_set, predictions))
}
