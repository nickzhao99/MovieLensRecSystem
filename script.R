
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")

# Load libraries
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(glmnet)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
```


nrow(edx) #returns number of rows
ncol(edx) #returns number of columns


#creates histogram of ratings
hist(edx$rating, main = "Histogram of ratings", xlab = "Rating", col ="Lightblue")

edx <- edx %>% group_by(userId) %>% mutate(usernumratings = n())
# creates the column that displays the user's total number of ratings given

edx <- edx %>% group_by(movieId) %>% mutate(movienumratings = n())
# creates the column that displays the movie's total number of ratings

edx <- edx %>% group_by(movieId) %>% mutate(averagemovierating = (sum(rating))/movienumratings)
# creates the column that displays the movie's average ratings

edx <- edx %>% group_by(userId) %>% mutate(averageuserrating = (sum(rating))/usernumratings)
# creates the column that displays the user's average ratings given

hist(edx$movienumratings, main = "Histogram of # of Ratings for Movies",col="lightblue")
# creates histogram for number of ratings for movies

hist(edx$usernumratings, main = "Histogram of # of Ratings for Users", col="lightblue")
# creates histogram for number of ratings for users

mu <- mean(edx$usernumratings) # calcualtes mean
sdv <- sd(edx$usernumratings) # calculates standard deviation
edx <- edx %>% mutate(zusernumratings = (usernumratings-mu)/sdv)
# normalized usernumratings using the formula (xi-mu)/sd

mu <- mean(edx$movienumratings) # calcualtes mean
sdv <- sd(edx$movienumratings) # calculates standard deviation
edx <- edx %>% mutate(zmovienumratings = (movienumratings-mu)/sdv)

# normalized movienumratings using the formula (xi-mu)/sd

y <- edx$rating # creates outcome vector
x <- edx %>% select(zusernumratings,averageuserrating,zmovienumratings,averagemovierating) %>% data.matrix() 
# creates input matrix with 5 variables, the userId is automatically used.


cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = (c(0.465,0.475,0.047)))
# finds the best lambda penalty rate for x input, y output, and alpha of 0 to indicate 
#a ridge regression using the glmnet package. The original method checks over the range of 
#-2 and 5, but is now reduced to save significant amounts of time from hours to seconds.


cv_fit$lambda.min # displays the best lambda penalty rate

model <- glmnet(x, y, alpha = 0, lambda = cv_fit$lambda.min)
#fits the model of x input matrix, y output vector, alpha of 0 to indicate ridge regression, and using the best lambda penalty rate found.

validation <- validation %>% group_by(userId) %>% mutate(usernumratings = n())
# creates usernumrating column

validation <- validation %>% group_by(movieId) %>% mutate(movienumratings = n())
# creates movienumratings column

validation <- validation %>% group_by(movieId) %>% mutate(averagemovierating = (sum(rating))/movienumratings)
# creates average movie rating column

validation <- validation %>% group_by(userId) %>% mutate(averageuserrating = (sum(rating))/usernumratings)
# calcualtes and creates average rating given by the user

mu <- mean(validation$usernumratings)
sdv <- sd(validation$usernumratings)
validation <- validation %>% mutate(zusernumratings = (usernumratings-mu)/sdv)
# normalizes usernumratings

mu <- mean(validation$movienumratings)
sdv <- sd(validation$movienumratings)
validation <- validation %>% mutate(zmovienumratings = (movienumratings-mu)/sdv)
#normalizes movienumratings

test <- validation %>% select(zusernumratings,averageuserrating,zmovienumratings,averagemovierating) %>% data.matrix()
#makes a matrix for x variables


predictions <- model %>% predict(test) %>% as.vector()
# predicts the test matrix using our model and outputs a vector.

data.frame(
     RMSE = RMSE(predictions, validation$rating),
     Rsquare = R2(predictions, validation$rating))

# Calculates the RMSE and R-squared value using predictions and rating in the validation dataset.

RMSE = RMSE(predictions, validation$rating)
# reports RMSE
RMSE


## Conclusion
#The RMSE is 0.844. Below the 0.865 requirement. I learned a great deal about machine learning through this project. I also spent a lot of time creating many instances of different algorithms before finding the one that works. The best advice for myself is to know the strength and weaknesses of each algorithms before spending hours of time running them. 
