![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

This project was part of a 9-week training course I attended in 2024.
- **Bootcamp**: Ironhack [AI Engineering bootcamp](https://www.ironhack.com/de-en/artificial-intelligence/remote)
- **Date**: September to November 2024
- **Project topics**: Data processing, Clustering, Sentiment Analysis, Generative AI, Prompt FineTuning, Sklearn, LLMs, Mistral7B

Final Grade and teacher's feedback:
```
- Presentation a bit rushed, you spent 3 minutes just with the intro, and the interesting parts were at the end.
- Great choices, very smart decisions and combination of techniques
- Excellent visual evaluation of the clustering model, which was probably the hardest one as there were no real labels
- Preprocessing and modeling was very good, with lots of experimentation
- Good README
- The code is well structured in notebooks, with clear sections, and very readable

Grade: 11.40 / 12
```

----

# Robo Reviews - Automated Product Review Generation

Robo Reviews is a project designed to process, analyze, and summarize product reviews. It leverages transformer models for sentiment analysis, clustering models to group products into categories, and a text generation model to produce insightful product reviews based on recurring user feedback. The tool provides insights into the most frequently mentioned pros and cons, helping users quickly understand the strengths and weaknesses of various products.

<img src="readme/robo_reviews.png" width=500>

## Project Setup
Follow these steps to clone the repository and create a virtual environment:

```bash
git clone https://github.com/alexdjulin/ik_robo_reviews.git
cd ik_robo_reviews
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## Files Description

### Notebooks
- **1_dataset_review.ipynb:** Jupyter notebook for initial dataset exploration and review analysis. Includes preprocessing of the review data.
- **2a_category_clustering_SimilarityClustering.ipynb:** Notebook implementing similarity-based clustering to categorize products based on review embeddings.
- **2b_category_clustering_UnsupervisedLearning.ipynb:** Notebook for unsupervised clustering methods, including KMeans, to categorize products into specific groups.
- **3_sentiment_analysis_model.ipynb:** Notebook exploring models fine-tuned on review sentiments. Then fine-tuning a DistilBERT model for sentiment analysis and evaluates the model’s performance on the dataset.
- **4_products_score.ipynb:** Notebook scoring products based on the number of positive, neutral, and negative reviews, and to identify the best products per category.
- **5_generator.ipynb:** Notebook exploring prompt fine-tuning on Mistral 7B to construct a structured product description from user reviews.
- **demo1_gradio.ipynb:** Gradio demo for clustering and sentiment analysis models.
- **demo2_review.ipynb:** Gradio demo for summarizing product reviews based on input data, showcasing the model’s review summarization capabilities.

### Custom Modules
- **helpers.py:** A helper module that contains various utility functions to load data, preprocess text, and interact with models.
- **prompting.py:** Contains code for generating prompts for the models, including text generation and summarization prompts.

### Documents
- **robo_reviews_presentation.pdf:** Final presentation of the project