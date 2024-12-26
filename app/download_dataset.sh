#!/bin/bash
curl -L -o ./source.zip https://www.kaggle.com/api/v1/datasets/download/grouplens/movielens-20m-dataset
unzip source.zip -d ./source/