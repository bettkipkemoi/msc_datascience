library(dplyr)
#iris dataset available on r by default
View(iris)
#arranging data
iris_arr <- iris %>% arrange(desc(Sepal.Width))
View(iris_arr)

#filtering data
iris_filt <- iris %>% filter(Petal.Length == 4.0)
View(iris_filt)

#mutating data
iris_mut <- iris %>% 
  mutate(petal_perimeter = 2*Petal.Length + 2*Petal.Width)
View(iris_mut)

#rename columns
iris_rename <- iris_mut %>% 
  rename(Petal.Perimeter = petal_perimeter)
View(iris_rename)

#selecting columns
iris_slct <- iris_rename %>% 
  select(Sepal.Length, Sepal.Width, Petal.Perimeter)
View(iris_slct)

#slicing up
iris_slice <- iris_slct %>% 
  slice(1:15)
View(iris_slice)
