# Student Exam Score Predictor
A small machine learning project that predicts a student's exam score from their homework score and attendance, with an interactive web app to try it out.

## Why I built this
I started with the basics (lists, dictionaries, functions, files and classes) and grew this project step by step until it became a trained model inside a working web app.
My goal wasn't just to call a library. I wanted to understand what happens underneath, so I first wrote linear regression **from scratch in plain Python**, and only then used scikit-learn and checked that both gave the same answer.

## What the app does

Predict: move two sliders (homework and attendance) and instantly see the predicted exam score, the letter grade, and whether the student is likely to pass.
Class overview: explore the class with a table, summary numbers and a grade distribution chart.
How it works: see the exact formula the model learned, in plain language.
Bring your own data: upload your own CSV of students and the model retrains on it.

## Results

I compared three models on 150 students. To get an honest score, each model is tested on students it **did not** see during training, using cross-validation.

Both real models cut the error roughly in half compared with simple guessing, so they've genuinely learned something.

## Built with

Python · scikit-learn · pandas · Streamlit · Altair · pytest · GitHub Actions

## A note on the data

All student names and scores are **made up** for learning purposes. No real student data is used.

