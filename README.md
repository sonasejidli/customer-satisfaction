# Grocery Product Sentiment Analysis with BERT
Project Overview
This project focuses on Sentiment Analysis of grocery product reviews using Deep Learning. The objective is to classify unstructured customer feedback as Positive or Negative by fine-tuning a BERT (Bidirectional Encoder Representations from Transformers) model.
The project tackles the challenge of imbalanced datasets in a real-world scenario, evolving from data cleaning with Regular Expressions to model optimization using Threshold Moving.

Key Features
Regex Data Extraction: Extracted numerical ratings from raw text strings (e.g., "Rated 5.0 out of 5 stars") to create ground-truth labels.
Handling Imbalance: Applied Random Oversampling to the minority class (Negative reviews) to prevent the model from ignoring critical feedback.
BERT Fine-Tuning: Customized the bert-base-uncased architecture for binary classification.
Dynamic Thresholding: Adjusted the classification threshold (from default 0.5 to 0.8) to improve the detection of negative reviews.
Local Inference: Includes a script for testing the model on custom sentences.

Data Preprocessing & Regex
One of the critical steps in this project was structuring the raw data.
Problem: The rating column contained strings like "Rated 4.0 out of 5 stars".
Solution (Regex): Used Python's re module to extract the floating-point number.

Labeling:
4.0 - 5.0 Positive (1)
1.0 - 3.0  Negative (0)

Model Performance
The model was fine-tuned for 2 epochs. Due to the high imbalance (Positive >> Negative), standard accuracy is not the only metric to consider.
Metric	        Result	            Insight
Accuracy	      81%	                The model generalizes well on the majority class.
F1-Score
(Positive)      0.89	              High reliability for positive feedback.
Recall 
(Negative)	    Improved	            Initially low, but improved by adjusting the decision threshold to 0.8, forcing the model to be "stricter" on potential negative reviews.

Tech Stack
Language: Python

Deep Learning: PyTorch, Transformers (Hugging Face)

Data Manipulation: Pandas, NumPy, Regex (re)

Visualization: Matplotlib, Seaborn (Confusion Matrix & EDA)


Future Improvements
Data Collection: The primary limitation is the scarcity of unique negative reviews. Collecting more negative data is essential.

Advanced Augmentation: Implementing synonym replacement or back-translation to diversify the minority class.

Deployment: Deploying the model as an API using FastAPI or a public Gradio Space.






