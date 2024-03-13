source("deepsurv.R")

# BiocManager::install("rhdf5")
X_train <- rhdf5::h5read("Data/simulated_linear.h5", "train/x") %>% t()
event_train <- rhdf5::h5read("Data/simulated_linear.h5", "train/e") %>% c()
time_train <- rhdf5::h5read("Data/simulated_linear.h5", "train/t") %>% c()
Y_train <- cbind(event_train, time_train)

X_test <- rhdf5::h5read("Data/simulated_linear.h5", "test/x") %>% t()
event_test <- rhdf5::h5read("Data/simulated_linear.h5", "test/e") %>% c()
time_test <- rhdf5::h5read("Data/simulated_linear.h5", "test/t") %>% c()
Y_test <- cbind(event_test, time_test)

m1 <- colMeans(X_train)
s1 <- apply(X_train, 2, sd)
X_train <- scale(X_train, center = m1, scale = s1)
X_test <- scale(X_test, center = m1, scale = s1)

n_patients_train <- dim(X_train)[1]
n_features <- dim(X_train)[2]

model = build_deepsurv(num_input=n_features,
                       num_layer= 1,
                       string_activation ="selu",
                       num_nodes = 4,
                       num_lr = 2.922e-4,
                       num_l2 = 1.999,
                       num_dropout = 0.375,
                       lr_decay = 3.579e-4)

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


print(c_index(LP_train, Y_train[, 2], Y_train[, 1]))
print(c_index(LP_test, Y_test[, 2], Y_test[, 1]))
