# Ghouls Goblins Ghosts EDA

# data imputation

library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
# read in data
train <- vroom("train.csv")
test <- vroom("test.csv")



library(lightgbm)
library(bonsai)
boost_recipe <- recipe(type~., data=train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())




boost_mod <- boost_tree(trees= tune(), tree_depth = tune(),
                        learn_rate = tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("lightgbm")


boost_wf <- 
  workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost_mod)

## set up a tuning grid
tuning_grid <-
  grid_regular(trees(),
               tree_depth(),
               learn_rate(),
               levels = 10)

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
  finalize_workflow(best_tune) %>%
  fit(data = train)

boost_preds <-
  final_wf %>%
  predict(new_data = test, type = "class")

# prepare and export preds to csv for kaggle

boost_output <- tibble(id = test$id, type = boost_preds$.pred_class)


vroom_write(boost_output, "GGG_BoostPreds.csv", delim = ",")

