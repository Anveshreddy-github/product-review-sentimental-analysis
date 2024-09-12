from flask import Flask, request, render_template, send_file
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import nltk
import re
import string
import base64
from io import BytesIO
from fpdf import FPDF

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/page2', methods=['GET'])
def page2():
    product_type = request.args.get('product-type')
    sample_size = request.args.get('sample-size')
    return render_template('page2.html', sample_size=sample_size, product_type=product_type)

@app.route('/analyze', methods=['POST'])
def analyze():
    product_type = request.form.get('product-type')
    sample_size = request.form.get('sample-size')
    if sample_size is None or product_type is None:
        return "Sample size or product type not provided", 400

    try:
        sample_size = int(sample_size)
    except ValueError:
        return "Invalid sample size", 400

    reviews = [request.form.get(f'review{i}') for i in range(1, sample_size + 1)]
    reviews = [review for review in reviews if review]

    if not reviews:
        return "Please provide at least one review.", 400

    data = pd.DataFrame(reviews, columns=["Review"])
    pie_data, wordcloud_img = process_and_analyze(data)
    
    recommendation = generate_recommendation(product_type, data)

    return render_template('result.html', pie_data=pie_data, wordcloud_img=wordcloud_img, recommendation=recommendation, product_type=product_type)

def process_and_analyze(data):
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    stopword = set(nltk.corpus.stopwords.words('english'))
    stemmer = nltk.SnowballStemmer("english")
    
    def clean(text):
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stopword]
        text = " ".join(text)
        text = [stemmer.stem(word) for word in text.split(' ')]
        text = " ".join(text)
        return text
    
    data["Review"] = data["Review"].apply(clean)
    
    sentiments = SentimentIntensityAnalyzer()
    data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Review"]]
    data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Review"]]
    data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Review"]]
    
    ratings = data[["Positive", "Negative", "Neutral"]].mean()
    pie_data = [dict(labels=ratings.index.tolist(), values=ratings.values.tolist(), type='pie', hole=.5)]
    
    text = " ".join(i for i in data.Review)
    wordcloud = WordCloud(stopwords=stopword, background_color="white").generate(text)
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    wordcloud_img = base64.b64encode(img.getvalue()).decode()
    
    return pie_data, wordcloud_img

def generate_recommendation(product_type, data):
    avg_sentiment = data[['Positive', 'Negative', 'Neutral']].mean()
    sentiment_score = avg_sentiment['Positive'] - avg_sentiment['Negative']

    if product_type in ['fashion', 'household', 'movies']:
        if sentiment_score > 0.1:
            return f"For {product_type}, it is recommended to proceed."
        else:
            return f"For {product_type}, it is recommended to avoid."
    elif product_type in ['flipkart', 'amazon']:
        if sentiment_score > 0.1:
            return "Based on reviews, the product is good to buy."
        else:
            return "Based on reviews, it is advisable to reconsider buying the product."
    elif product_type in ['zomato', 'swiggy']:
        if sentiment_score > 0.1:
            return "The food is recommended based on reviews."
        else:
            return "The food may not be up to expectations according to reviews."
    elif product_type == 'restaurants':
        if sentiment_score > 0.1:
            return "The restaurant is recommended based on reviews."
        else:
            return "It may be wise to consider other options."
    elif product_type == 'others':
        if sentiment_score > 0.1:
            return "The product/service is recommended."
        else:
            return "Consider other options based on the reviews."
    else:
        return "No specific recommendation available for this product type."

@app.route('/download-pdf')
def download_pdf():
    product_type = request.args.get('product-type')
    reviews = request.args.getlist('review')
    
    if not product_type or not reviews:
        return "Product type or reviews not provided", 400

    data = pd.DataFrame(reviews, columns=["Review"])
    recommendation = generate_recommendation(product_type, data)
    pie_data, _ = process_and_analyze(data)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Product Reviews Sentiment Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Product Type: {product_type}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt="Recommendation: " + recommendation)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Reviews:", ln=True)
    for review in reviews:
        pdf.multi_cell(0, 10, txt=review)
        pdf.ln(5)
    
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return send_file(pdf_output, as_attachment=True, download_name='analysis_report.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)
