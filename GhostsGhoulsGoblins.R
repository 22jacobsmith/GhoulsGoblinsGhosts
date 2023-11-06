# GhostGhoulsGoblins.R
# libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
# read in data
train <- vroom("train.csv")
test <- vroom("test.csv")
imputed_set <- vroom("trainWithMissingValues.csv")
#head(train)

# impute the missing values in the data set

### DATA IMPUTATION
impute_recipe_mean <- recipe(type~., data=imputed_set) %>%
  step_impute_mean(all_numeric_predictors())

prep <- prep(impute_recipe_mean)
baked <- bake(prep, new_data = imputed_set)

rmse_vec(train[is.na(imputed_set)], baked[is.na(imputed_set)])

impute_recipe_knn <-
  recipe(type~., data=imputed_set) %>%
  step_impute_knn(bone_length, impute_with = imp_vars(c('has_soul')), neighbors = 3) %>%
  step_impute_knn(rotting_flesh, impute_with = imp_vars(c('has_soul', 'bone_length')), neighbors = 3) %>%
  step_impute_knn(hair_length, impute_with = imp_vars(c('has_soul', 'bone_length', 'rotting_flesh')), neighbors = 3)

prep <- prep(impute_recipe_knn)
baked <- bake(prep, new_data = imputed_set)

rmse_vec(train[is.na(imputed_set)], baked[is.na(imputed_set)])


impute_recipe_linear <-
  recipe(type~., data=imputed_set) %>%
  step_impute_linear(bone_length, impute_with = c('has_soul', 'type')) %>%
  step_impute_linear(rotting_flesh, impute_with = c('has_soul', 'type', 'bone_length')) %>%
  step_impute_linear(hair_length,
                     impute_with = c('has_soul', 'type', 'bone_length', 'rotting_flesh'))


prep <- prep(impute_recipe_linear)
baked <- bake(prep, new_data = imputed_set)

rmse_vec(train[is.na(imputed_set)], baked[is.na(imputed_set)])
# best imputation RMSE: Linear, 0.1312063

impute_recipe_bag <-
  recipe(type~., data=imputed_set) %>%
  step_impute_bag(bone_length, impute_with = c('has_soul', 'type'), trees = 500) %>%
  step_impute_bag(rotting_flesh, impute_with = c('has_soul', 'type', 'bone_length'), trees = 500) %>%
  step_impute_bag(hair_length,
                     impute_with = c('has_soul', 'type', 'bone_length', 'rotting_flesh'), trees = 500)



prep <- prep(impute_recipe_bag)
baked <- bake(prep, new_data = imputed_set)

rmse_vec(train[is.na(imputed_set)], baked[is.na(imputed_set)])


## MODEL FITTING ##
## RANDOM FOREST

rf_recipe <-
  recipe(type~., data=train) %>%
  step_normalize(all_numeric_predictors())

rf_mod <- rand_forest(min_n = tune(), mtry = tune()) %>%
  set_engine('ranger') %>%
  set_mode('classification')


rf_wf <-
  workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_mod)



## set up a tuning grid
tuning_grid <-
  grid_regular(mtry(range = c(1,6)),
               min_n(),
               levels = 6)

## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  rf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  rf_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train)

rf_preds <-
  final_wf %>%
  predict(new_data = test, type = 'class')

# prepare and export preds to csv for kaggle

rf_output <- tibble(id = test$id, type = rf_preds$.pred_class)

vroom_write(rf_output, "GGG_RFPreds.csv", delim = ",")

## KNN

knn_recipe <-
  recipe(type~., data=train) %>%
  step_normalize(all_numeric_predictors())

knn_mod <- nearest_neighbor(neighbors = tune(), dist_power = tune()) %>%
  set_engine('kknn') %>%
  set_mode('classification')


knn_wf <-
  workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_mod)


## set up a tuning grid
tuning_grid <-
  grid_regular(neighbors(),
               dist_power(),
               levels = 6)

## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  knn_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train)

knn_preds <-
  final_wf %>%
  predict(new_data = test, type = 'class')

# prepare and export preds to csv for kaggle

knn_output <- tibble(id = test$id, type = knn_preds$.pred_class)

vroom_write(knn_output, "GGG_KNNPreds.csv", delim = ",")


## NAIVE BAYES

library(discrim)

nb_recipe <-
  recipe(type~., data=train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold = .99)

nb_mod <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_engine('naivebayes') %>%
  set_mode('classification')


nb_wf <-
  workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_mod)


## set up a tuning grid
tuning_grid <-
  grid_regular(Laplace(),
               smoothness(),
               levels = 6)

## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  nb_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train)

nb_preds <-
  final_wf %>%
  predict(new_data = test, type = 'class')

# prepare and export preds to csv for kaggle

nb_output <- tibble(id = test$id, type = nb_preds$.pred_class)

vroom_write(nb_output, "GGG_NBPreds.csv", delim = ",")

### SVM


svm_recipe <- recipe(type~., data=train) %>%
   step_normalize(all_numeric_predictors())




svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")


svm_wf <- 
  workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svmRadial)

## set up a tuning grid
tuning_grid <-
  grid_regular(rbf_sigma(),
               cost(),
               levels = 20)

## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  svm_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  svm_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train)

svm_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")

# prepare and export preds to csv for kaggle

svm_output <- tibble(id = test$id, type = svm_preds$.pred_class)


vroom_write(svm_output, "GGG_SVMPreds.csv", delim = ",")



### boosting


boost_recipe <- recipe(type~., data=train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())




boost_mod <- boost_tree(trees= 2000, tree_depth = 4,
                        learn_rate = .000562) %>% # set or tune
  set_mode("classification") %>%
  set_engine("xgboost")


boost_wf <- 
  workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost_mod)

## set up a tuning grid
tuning_grid <-
  grid_regular(trees(),
               tree_depth(),
               learn_rate(),
               levels = 5)

## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  boost_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")

# finalize wf and get preds

final_wf <-
  boost_wf %>%
  #finalize_workflow(best_tune) %>%
  fit(data = train)

boost_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")

# prepare and export preds to csv for kaggle

boost_output <- tibble(id = test$id, type = boost_preds$.pred_class)


vroom_write(boost_output, "GGG_BoostPreds.csv", delim = ",")



### NEURAL NETWORK

nn_recipe <- recipe(type~., data = train) %>%
update_role(id, new_role="id") %>%
step_mutate(color = as.factor(color)) %>% ## Turn color to factor then dummy encode color
step_dummy(color) %>%
 step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
set_engine("nnet") %>% #verbose = 0 prints off less
set_mode("classification")

nn_wf <- 
  workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe)


nn_tuning_grid <- grid_regular(hidden_units(range=c(1, 30)),
                            levels=30)


## split into folds
folds <- vfold_cv(train, v = 5, repeats = 1)

# run cv

CV_results <-
  nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tuning_grid,
            metrics = metric_set(accuracy))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("accuracy")



# finalize wf and get preds

final_wf <-
  nn_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train)

nn_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")

# prepare and export preds to csv for kaggle

nn_output <- tibble(id = test$id, type = nn_preds$.pred_class)


vroom_write(nn_output, "GGG_NNPreds.csv", delim = ",")

## view graphic output

tuned_nn <-
  nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tuning_grid,
            metrics = metric_set(accuracy))

tuned_nn %>% collect_metrics() %>%
filter(.metric=="accuracy") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()


# x axis - hidden units
# y axis - accuracy (mean)
