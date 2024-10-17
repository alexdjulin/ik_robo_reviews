#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File Name: prompting.py
Author: Alexandre Donciu-Julin
Date: 2024-10-17
Description: This file contains the prompts and functions used with Mistral to infer on the dataset.
"""

# Import statements --------------------------------------------------------------------------------
import re
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants ----------------------------------------------------------------------------------------
SEP = 100 * '-'

# PROMPT 1 -----------------------------------------------------------------------------------------
# Summarize an Amazon review into a concise sentence
prompt_review_summary = """
Summarize the following product review information. Return an empty string if you are unable to generate a summary.

Review 1:
"This laptop exceeded my expectations. The battery lasts all day, and it’s super lightweight, which makes it perfect for traveling. The display is clear and bright, and the performance is fast even with multiple programs running. My only complaint is that the keyboard feels a bit shallow. Overall, I’m really happy with my purchase!"
Summary 1:
->"Lightweight and fast laptop with a great display and long-lasting battery, but the keyboard is a bit shallow."

Review 2:
"The headphones are okay but not as good as I expected. The sound quality is decent, and they’re comfortable to wear for a while. However, they don’t block out background noise as well as I hoped. For the price, I think there are better options out there."
Summary 2:
->"Decent sound quality and comfortable, but poor noise isolation and better options exist for the price."

Review 3:
{review_text}
Summary 3:
"""

# PROMPT 2 -----------------------------------------------------------------------------------------
# Extract a list of recurring ideas from a set of summarized reviews
prompt_reviews_recurring_ideas = """
Analyze the following list of product reviews. Provide a concise summary of the three most frequently mentioned ideas or themes. Ensure each point reflects common feedback without repeating phrasing.

### Reviews:
[Review 1]: "The camera quality is amazing and takes very clear photos, even in low light."
[Review 2]: "Battery life could be better; it drains faster than my previous phone."
[Review 3]: "I’m impressed with the screen clarity and brightness, but the battery doesn’t last very long."
[Review 4]: "The photos are sharp and vibrant, especially in daylight."
[Review 5]: "Battery life is a bit disappointing, but the display is excellent and great for watching videos."

### Recurring Ideas:
1. The camera quality is highly praised, especially for clear and vibrant photos in various lighting.
2. The display quality is appreciated for its clarity and brightness, making it ideal for media.
3. Battery life is a common complaint, with users noting that it drains quickly.

### Reviews:
[Review 1]: "The sound quality on these headphones is outstanding, with deep bass and clear highs."
[Review 2]: "They’re comfortable to wear for long hours, but I wish they blocked out noise better."
[Review 3]: "Amazing audio clarity, but they don’t fully cancel out background sounds."
[Review 4]: "Great sound quality for the price, but I can still hear some outside noise."
[Review 5]: "Comfortable fit and good audio, but not ideal for noisy environments."

### Recurring Ideas:
1. Sound quality is praised for its clarity and depth, especially given the price.
2. Comfort is a major positive, with users finding them suitable for extended wear.
3. Noise cancellation is lacking, with multiple reviews mentioning that they allow outside noise.

### Reviews:
{review_text}

### Recurring Ideas:
"""

# PROMPT 3 -----------------------------------------------------------------------------------------
# Generate a product review with a title and product description based on a list of recurring ideas
prompt_final_product_review = """
Write a positive product review from these recurring ideas, that presents the product, highlights its pros, and subtly mentions a few minor downsides reported by some users.

### Product Name:
Experience V2 Smartphone

### Recurring ideas:
Good battery life, enough for a full day of use.
Sturdy and premium design.
Good camera performance, clear, high-quality photos, especially in good lighting.
Price slightly high for the features offered.

### Positive product review:
This product offers excellent battery life, lasting a full day, and has a sturdy, premium design.
The camera performs well, capturing clear photos, especially in good lighting.
Though slightly pricey for the features, its quality build and reliability make it a solid choice.

### Product Name:
{product_name}

### Recurring ideas:
{review_ideas}

### Positive product review:
"""


# PROMPTING FUNCTIONS ------------------------------------------------------------------------------
# The following functions use the above prompts to generate text based on the provided input.

def load_model_and_tokenizer(model_name: str = "mistralai/Mistral-7B-v0.3") -> tuple:
    """Load the Mistral model and tokenizer for inference.

    Args:
        model_name (str, optional): Model name. Defaults to "mistralai/Mistral-7B-v0.3".

    Returns:
        tuple: Model and tokenizer objects.
    """

    # load llama tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    # load 4bit quantization model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto"
    )

    return model, tokenizer


def run_inference_on_model(model: object, tokenizer: object, prompt: str, max_tokens: int = 100) -> str:
    """Run inference on the model to generate text based on the provided prompt.

    Args:
        model (transformers.models.mistral.modeling_mistral.MistralForCausalLM): model instance
        tokenizer (transformers.models.llama.tokenization_llama.LlamaTokenizer): tokenizer instance
        prompt (str): input prompt for the model
        max_tokens (int, optional): maximum number of tokens to generate. Defaults to 100.

    Returns:
        str: generated text based on the model's output
    """

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    ).to(device)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_review_summary(model: object, tokenizer: object, review_text: str, max_tokens=50) -> str:
    """Run inference on the model to generate a review summary based on the provided review.

    Args:
        model (transformers.models.mistral.modeling_mistral.MistralForCausalLM): model instance
        tokenizer (transformers.models.llama.tokenization_llama.LlamaTokenizer): tokenizer instance
        review_text (str): review text to summarize

    Returns:
        str: review summary extracted from the model's answer using a regex pattern
    """

    # run inference
    prompt = prompt_review_summary.format(review_text=review_text)
    result = run_inference_on_model(model, tokenizer, prompt, max_tokens)

    # Match all summaries marked by '->"' and capture their content
    match = re.findall(r'->"(.*?)"', result, re.DOTALL)
    # Return the third summary with leading/trailing spaces removed
    if match and len(match) >= 3:
        summary = match[2].strip()
    else:
        # print("No review summary found.")
        summary = ""

    return summary


def generate_reviews_recurring_ideas(model: object, tokenizer: object, review_text: str, max_tokens: int = 100) -> str:
    """Run inference on the model to generate recurring ideas from a set of summarized reviews.

    Args:
        model (transformers.models.mistral.modeling_mistral.MistralForCausalLM): model instance
        tokenizer (transformers.models.llama.tokenization_llama.LlamaTokenizer): tokenizer instance
        review_text (str): review text to analyze
        max_tokens (int, optional): maximum number of tokens to generate. Defaults to 100.

    Returns:
        str: recurring ideas extracted from the model's answer using a regex pattern
    """

    # run inference on model
    prompt = prompt_reviews_recurring_ideas.format(review_text=review_text)
    result = run_inference_on_model(model, tokenizer, prompt, max_tokens)

    # extract recurring ideas
    recurrent_ideas = result.split('### Recurring Ideas:\n')[-1].split('\n')
    # remove any empty strings or text after the last ideas
    recurrent_ideas = recurrent_ideas[:recurrent_ideas.index("")] if "" in recurrent_ideas else recurrent_ideas
    # remove the last one in case it's incomplete
    if not recurrent_ideas[-1].endswith('.'):
        recurrent_ideas = recurrent_ideas[:-1]
    # remove the numbers that may confuse the model
    recurrent_ideas = [ideas[3:] for ideas in recurrent_ideas]

    return '\n'.join(recurrent_ideas)


def generate_final_review(model: object, tokenizer: object, product_name: str, review_ideas: str, max_tokens: int = 150) -> tuple:
    """Run inference on the model to generate a final product review based on a list of recurring ideas.

    Args:
        model (transformers.models.mistral.modeling_mistral.MistralForCausalLM): model instance
        tokenizer (transformers.models.llama.tokenization_llama.LlamaTokenizer): tokenizer instance
        product_name (str): name of the product (helps the model generate a title)
        review_ideas (str): recurring ideas to include in the review
        max_tokens (int, optional): maximum number of tokens to generate. Defaults to 150.

    Returns:
        tuple: title and review generated by the model
    """

    # run inference on model
    prompt = prompt_final_product_review.format(product_name=product_name, review_ideas=review_ideas)
    result = run_inference_on_model(model, tokenizer, prompt, max_tokens)

    # extract the review
    final_review = result.split('### Positive product review:\n')[-1]

    # remove line breaks
    final_review = final_review.replace('\n', ' ')

    # remove any trailing artifacts
    final_review = final_review.split('###')[0].strip()

    # remove last sentence from review if incomplete
    if not final_review.endswith('.'):
        final_review = '.'.join(final_review.split('.')[:-1]) + '.'

    # summarize the review to get title
    title = generate_review_summary(model, tokenizer, final_review, max_tokens=20)

    # set default title if missing or too long (first sentence of review)
    if len(title.split()) < 2 or len(title.split()) > 15:
        title = final_review.split('.')[0]

    return title, final_review
