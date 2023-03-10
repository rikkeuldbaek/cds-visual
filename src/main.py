import data_prep
import logreg_model

# Preprocess data
X_train, y_train, X_test, y_test, labels = data_prep.load_data_func()

X_train_grey, X_test_grey = data_prep.grey_scale_convert_func(X_train, X_test)
X_train_scaled, X_test_scaled = data_prep.scale_data_func()
X_train_dataset,X_test_dataset= data_prep.reshape_data_func()



# # Logistic regression model
logreg_model.logreg_model_function()

