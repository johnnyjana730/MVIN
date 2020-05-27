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
	bash main_az_book_mvin.sh $gpu
    elif [ $dataset = "movie" ]
    then
	bash main_movie_mvin.sh $gpu
    elif [ $dataset = "last_fm" ]
    then
	bash main_last-fm_mvin.sh $gpu
    else
	echo "Invalid dataset! Dataset should be 'amazon-book', 'movie', or 'last_fm'."
    fi
elif [ $model_type = "KGCN" ]
then 
    if [ $dataset = "amazon-book" ]
    then
	bash main_az_book_kgcn.sh $gpu
    elif [ $dataset = "movie" ]
    then
	bash main_movie_kgcn.sh $gpu
    elif [ $dataset = "last_fm" ]
    then
	bash main_last-fm_kgcn.sh $gpu
    else
	echo "Invalid dataset! Dataset should be 'amazon-book', 'movie', or 'last_fm'."
    fi 
elif [ $model_type = "RippleNet" ]
then 
    if [ $dataset = "amazon-book" ]
    then
	bash main_az_book_ripplenet.sh $gpu
    elif [ $dataset = "movie" ]
    then
	bash main_movie_ripplenet.sh $gpu
    elif [ $dataset = "last_fm" ]
    then
	bash main_last-fm_ripplenet.sh $gpu
    else
	echo "Invalid dataset! Dataset should be 'amazon-book', 'movie', or 'last_fm'."
    fi
else 
    echo "Invalid model!"
    exit 1
fi
