library(tidyverse)
library(jpeg)
library(raster)
library(caret)

image <- readImage("../Data/Images/POC/background_removed_highly_filtered.jpg")
image_255 <- image*255

training <- read_tsv("../Data/bayes_classes.tsv", col_types = "ccc") %>%
  pivot_longer(cols = everything(), names_to = "Pixel_Type", values_to = "RGB") %>%
  separate(col = RGB, into = c("Red", "Green", "Blue"), sep = ",") %>%
  mutate(Pixel_Type = factor(Pixel_Type),
         Red = as.numeric(Red),
         Green = as.numeric(Green),
         Blue = as.numeric(Blue))

set.seed(7)
nb_model <- train(Pixel_Type ~ .,
                  data = training,
                  method = "nb",
                  metric = "Accuracy",
                  trControl = trainControl(method = "cv",
                                           savePredictions = T, 
                                           allowParallel = T
                  )
)

red_chan <- channel(image_255, mode = "red") %>% # extracting 
  as_tibble() %>%
  mutate(row = row_number()) %>%
  pivot_longer(cols = -row, names_to = "column", values_to = "Red", names_prefix = "V")

green_chan <- channel(image_255, mode = "green") %>% # extracting 
  as_tibble() %>%
  mutate(row = row_number()) %>%
  pivot_longer(cols = -row, names_to = "column", values_to = "Green", names_prefix = "V")

blue_chan <- channel(image_255, mode = "blue") %>% # extracting 
  as_tibble() %>%
  mutate(row = row_number()) %>%
  pivot_longer(cols = -row, names_to = "column", values_to = "Blue", names_prefix = "V")

full_color <- red_chan %>%
  full_join(green_chan) %>%
  full_join(blue_chan)

predictions <- predict(nb_model, full_color %>%
                         head(100000))

full_color %>%
  head(100000) %>%
  mutate(Pixel_Type = predictions) %>%
  write_delim("../Outputs/Exploration/bayes_classes_by_pixel.txt")