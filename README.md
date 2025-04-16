# Predicting Financial Risk Using NLP: A Sentiment- and Entity-Based Approach

## Objective
This project investigates whether financial risk at the company level can be anticipated and quantified through sentiment analysis and named entity recognition (NER) in financial news. By analyzing recent headlines and articles from trusted sources such as Google News and Yahoo Finance, we aim to construct a sentiment-informed risk score and explore how it aligns with real market behavior, especially during periods of volatility.

The broader goal is to create a repeatable and scalable NLP framework that empowers both institutional and retail investors to gauge emerging risks in real time, driven by language signals in financial media.

## Research Question
To what extent can financial risk at the company level be anticipated and measured using sentiment analysis and named entity recognition in financial news, and how closely do these sentiment-based risk scores match the behavior of stock markets over time?

## Data Sources

News Headlines (Web-Scraped): Google News and Yahoo Finance headlines on selected companies (Tesla, Apple, Nvidia, Amazon) from the past 90 days.

Stock Prices: Historical daily price data from Yahoo Finance using yfinance, used to validate the sentiment-based risk signals with actual market behavior.

## Tools and Technologies

Python

Natural Language Processing Libraries:

spaCy for Named Entity Recognition (NER)

transformers for FinBERT sentiment classification

nltk and VADER for rule-based sentiment comparison

Data Handling & Visualization:

pandas, matplotlib, seaborn, tqdm

yfinance for financial data acquisition

## Model:

FinBERT (Pretrained transformer fine-tuned on financial sentiment data) & VADAR

## Methodology

## Data Collection:

News scraped in real-time using queries like “Tesla news,” “Amazon earnings,” etc.

Data filtered and stored in a preprocessed CSV format.

## Sentiment Analysis:

Applied both VADER (rule-based) and FinBERT (transformer-based) sentiment models.

Each news snippet is classified as positive, neutral, or negative.

Visual comparisons made to show how sentiment varies across companies and methods.

## Named Entity Recognition:

Used spaCy to identify entity types (ORG, GPE, DATE, etc.) across the corpus.

Mapped frequency of these entities to sentiment labels to analyze how news framing (via entity emphasis) influences risk perception.
