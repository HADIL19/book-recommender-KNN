# Book Recommendation Engine using K-Nearest Neighbors

A machine learning project built for the freeCodeCamp Machine Learning with Python certification. Given a book title, the model recommends 5 similar books based on how users rated them.

## What We Built

Using the **Book-Crossings dataset** (1.1 million ratings of 270,000 books by 90,000 users), we built a `get_recommends()` function that:

1. Takes a book title as input
2. Returns that title along with 5 similar books, each paired with a "distance" score (lower = more similar)

Example:
```python
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
```
returns the book title plus a list of 5 `[title, distance]` pairs for similar books.

## How It Works — Step by Step

### 1. Loading the data
Two CSVs were loaded: one mapping ISBNs to book titles/authors, and one containing individual user ratings (user ID, ISBN, rating 1–10).

### 2. Filtering for statistical significance
Most books in the raw dataset have only a handful of ratings, which makes similarity comparisons unreliable. We removed:
- **Users with fewer than 200 ratings** (occasional raters don't give enough signal about taste)
- **Books with fewer than 100 ratings** (obscure books can't be reliably compared)

This shrank the dataset dramatically but made every remaining rating more meaningful.

### 3. Building a user-item matrix
We reshaped the filtered ratings into a **pivot table**: one row per book, one column per user, and each cell holding that user's rating of that book (0 where a user never rated a book).

This turns every book into a numeric "vector" — its row of ratings across all users — which is what a distance-based algorithm needs to compare books.

We also had to **drop duplicate (user, title) pairs**, since a handful of titles are associated with more than one ISBN, which otherwise breaks the pivot operation.

Because this matrix is mostly zeros (a **sparse matrix**), we converted it into a `scipy.sparse.csr_matrix` — a memory-efficient format for matrices with mostly-empty cells.

### 4. Training the KNN model
We used `sklearn.neighbors.NearestNeighbors` with:
- `metric='cosine'` — cosine distance measures the angle between two rating vectors rather than their raw magnitude, which works better than Euclidean distance for sparse, high-dimensional rating data
- `algorithm='brute'` — exhaustively compares against all books, which is accurate and fast enough at this dataset size

The model doesn't "predict" a rating — it just learns the geometry of the rating space so it can find, for any given book, which other books have the most similar rating patterns across users.

### 5. Writing `get_recommends()`
Given a book title, the function:
1. Looks up that book's row in the pivot table
2. Asks the KNN model for its 6 nearest neighbors (5 real neighbors + itself, since a book is always closest to itself)
3. Returns the 5 neighbors and their distances, ordered from farthest to nearest — matching the expected output format

## Key Concepts Learned

- **K-Nearest Neighbors (KNN)** as an unsupervised similarity search, not just a classifier — here it's used to find "nearby" items in a rating-vector space rather than to predict a label.
- **Cosine similarity/distance** — why it outperforms Euclidean distance for sparse, high-dimensional data like ratings, since it cares about the *pattern* of ratings rather than their scale.
- **Data cleaning for statistical significance** — recognizing that including low-signal data (rarely-rated books, rarely-rating users) hurts model quality rather than helping it.
- **Pivot tables** — reshaping "long" transactional data (one row per rating) into "wide" matrix data (one row per book, one column per user), which is the standard input shape for most similarity/collaborative-filtering algorithms.
- **Sparse matrices** — why memory-efficient representations matter once data has hundreds of thousands of mostly-empty cells, and how `scipy.sparse` solves that.
- **Handling messy real-world data** — duplicate title/ISBN mappings had to be deduplicated before the pivot step, a reminder that real datasets rarely fit cleanly into the shape an algorithm expects on the first try.

## Files

- `fcc_book_recommendation_knn.py` — the completed solution code (filtering, pivot table, KNN model, and `get_recommends` function)
