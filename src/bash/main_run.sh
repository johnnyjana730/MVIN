#!/bin/bash

# bash main_movie.sh 1 &
# bash main_last-fm.sh 1 &
# bash main_az_book.sh 3 &

#bash main_att_case_st.sh 3 &

model_type=$1
dataset=$2
gpu=$3

if [ $model_type = "MVIN" ]
then 
    if [ $dataset = "amazon-book" ]
    then
	bash mvin_az_book.sh $gpu
    elif [ $dataset = "movie" ]
    then
	bash mvin_movie.sh $gpu
    elif [ $dataset = "last_fm" ]
    then
	bash mvin_last_fm.sh $gpu
    else
	echo "Invalid dataset! Dataset should be 'amazon-book', 'movie', or 'last_fm'."
    fi
elif [ $model_type = "KGCN" ]
then 
    if [ $dataset = "amazon-book" ]
    then
	bash kgcn_az_book.sh $gpu
    elif [ $dataset = "movie" ]
    then
	bash kgcn_movie.sh $gpu
    elif [ $dataset = "last_fm" ]
    then
	bash kgcn_last_fm.sh $gpu
    else
	echo "Invalid dataset! Dataset should be 'amazon-book', 'movie', or 'last_fm'."
    fi 
elif [ $model_type = "RippleNet" ]
then 
    if [ $dataset = "amazon-book" ]
    then
	bash ripplenet_az_book.sh $gpu
    elif [ $dataset = "movie" ]
    then
	bash ripplenet_movie.sh $gpu
    elif [ $dataset = "last_fm" ]
    then
	bash ripplenet_last_fm.sh $gpu
    else
	echo "Invalid dataset! Dataset should be 'amazon-book', 'movie', or 'last_fm'."
    fi
else 
    echo "Invalid model!"
    exit 1
fi
