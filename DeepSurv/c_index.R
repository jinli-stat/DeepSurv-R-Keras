library(Rcpp)

c_index <- function(y_true, y_pred, E){
  n <- length(y_true)
  numerator <- 0.0
  denominator <- 0.0
  
  is_comparable <- function(t_i, t_j, d_i, d_j){
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & (d_i | d_j))
  }
  is_concordant <- function(s_i, s_j, t_i, t_j, d_i, d_j){
    conc = 0.0
    if(t_i < t_j){
      conc <- (s_i < s_j) + (s_i == s_j) * 0.5
    }else if(t_i == t_j){
      if(d_i & d_j){
        conc <- 1.0 - (s_i != s_j) * 0.5
      }else if(d_i){
        conc <- (s_i < s_j) + (s_i == s_j) * 0.5
      }else if(d_j){
        conc <- (s_i > s_j) + (s_i == s_j) * 0.5
      }
    }
    return(conc * is_comparable(t_i, t_j, d_i, d_j))
  }
  
  for (i in 1:n) {
    for (j in 1:n){
      if(i!=j){
        numerator <- numerator + is_concordant(index[i, i], index[i, j], observed_time[i], observed_time[j], delta[i], delta[j])
      }
    }
  }
  
  for (k in 1:n) {
    for (m in 1:n) {
      if(k!=m){
        denominator = denominator + is_comparable(observed_time[k], observed_time[m], delta[k], delta[m])
      }
    }
  }
  
  return(numerator / denominator)
}