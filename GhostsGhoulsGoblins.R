# GhostGhoulsGoblins.R
# libraries
library(tidyverse)
library(tidymodels)
library(vroom)
# read in data
train <- vroom("train.csv")
test <- vroom("test.csv")
imputed_set <- vroom("trainWithMissingValues.csv")
head(train)

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
