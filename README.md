# Movie Genre Prediction via Multi-Label Classification

Predict movie genre tags from plot summaries using a Binary Relevance (One-vs-Rest) approach with TF-IDF features and Logistic Regression.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Environment Setup](#environment-setup)  
4. [Data Preparation & Preprocessing](#data-preparation--preprocessing)  
5. [Feature Extraction](#feature-extraction)  
6. [Modeling](#modeling)  
7. [Evaluation](#evaluation)  
8. [Inference](#inference)  
9. [Results](#results)  
10. [Future Improvements](#future-improvements)  
11. [References](#references)  

---

## Project Overview

This repository implements a multi-label classification pipeline to predict movie genres directly from plot summaries. We use the CMU Movie Summary Corpus and treat each plot as an instance that can have one or more genres.  

Key steps:
- Text cleaning & stopword removal  
- TF-IDF vectorization  
- One-vs-Rest Logistic Regression  
- Threshold tuning for optimal F1 score  

---

## Dataset

- **movie.metadata.tsv**  
  - Metadata for 81,741 movies (Freebase, Nov 2012)  
  - Contains `movie_id`, `movie_name` and `genre` fields  

- **plot_summaries.txt**  
  - Plot summaries for 42,306 English-language movies (Wikipedia, Nov 2012)  
  - Each line: `<movie_id>\t<plot summary>` 
