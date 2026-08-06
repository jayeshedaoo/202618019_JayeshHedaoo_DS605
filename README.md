# 202618019_JayeshHedaoo_DS605
# DS605 Lab 1 – Web Scraping, Data Cleaning and Data Analysis

## Student Information

- **Name:** Jayesh Hedaoo
- **Student ID:** 202618019
- **Course:** DS605
- **Assignment:** Lab 1

---

## Project Overview

This project demonstrates a complete data analytics pipeline starting from web scraping to data analysis.

The dataset was collected from the demo website **Books to Scrape** using the Scrapy framework. After scraping, the data was cleaned, preprocessed, visualized, and analyzed using Python libraries.

---

## Objectives

- Scrape book information using Scrapy
- Clean and preprocess the scraped data
- Perform exploratory data analysis (EDA)
- Generate visualizations
- Create a Word Cloud from book descriptions
- Draw insights from the dataset

---

## Technologies Used

- Python 3
- Scrapy
- Pandas
- NumPy
- Matplotlib
- Seaborn
- WordCloud
- Jupyter Notebook

---

## Project Structure

```
scraper/
│
├── notebooks/
│   ├── DS605_Lab1.ipynb
│   └── books_cleaned.csv
│
├── scraper/
│   ├── spiders/
│   │   └── books_spider.py
│   ├── items.py
│   ├── pipelines.py
│   ├── middlewares.py
│   ├── settings.py
│   └── books_raw.csv
│
├── scrapy.cfg
└── README.md
```

---

## Dataset Information

The dataset contains information about books including:

- Title
- Category
- Price
- Rating
- Availability
- Product Description
- UPC
- Number of Reviews
- Product URL

---

## Data Preprocessing

The following preprocessing steps were performed:

- Removed currency symbols from prices
- Converted prices into numeric format
- Converted ratings into numerical values
- Checked for missing values
- Exported cleaned dataset

---

## Visualizations

The notebook contains:

- Price Distribution
- Rating Distribution
- Category-wise Book Count
- Price vs Rating Scatter Plot
- Word Cloud of Product Descriptions

---

## Key Findings

- The dataset contains 105 books with no missing values.
- Most books are rated between 2 and 4 stars.
- The average price of books is approximately £34.6.
- Some categories contain considerably more books than others.
- Price and rating show only a weak relationship.

---

## Limitations

- The dataset contains only 105 books.
- Ratings are represented as star values.
- Sales and popularity information are unavailable.
- The data comes from a demo website and may not represent real-world bookstores.

---

## Repository Contents

- Scrapy Project
- Jupyter Notebook
- Raw Dataset
- Cleaned Dataset
- Visualizations
- Word Cloud
- Analysis
- README

---

## Source

Website used for scraping:

https://books.toscrape.com/