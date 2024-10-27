import finnhub
import json
from datetime import datetime, timedelta
import os
from groq import Groq
import requests
from requests.exceptions import RequestException
import pandas as pd
import mplfinance as mpf
import base64
import time
import sys
from pyuploadcare import Uploadcare

# Global variable to store the folder name
SAVE_FOLDER_NAME = None

class StockCandlestickChart:
    def __init__(self, api_key, stock_symbol, interval, save_location):
        # Initialize with necessary attributes
        self.api_key = api_key
        self.stock_symbol = stock_symbol
        self.interval = interval
        self.save_location = save_location
        self.df = None

    def fetch_data(self):
        # Fetch the intraday data from Alpha Vantage API
        print('Stock Symbol to be called is - ' + self.stock_symbol)
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={self.stock_symbol}&interval={self.interval}&apikey={self.api_key}'
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Error fetching data from Alpha Vantage API: {response.status_code}")
        data = response.json()

        # Check if the response contains an error message
        if "Note" in data:
            raise ValueError("API call frequency limit reached. Please wait and try again.")
        if "Error Message" in data:
            raise ValueError(f"Error message from Alpha Vantage API: {data['Error Message']}")

        # Extract the time series data based on the provided interval
        time_series_key = f'Time Series ({self.interval})'
        time_series = data.get(time_series_key, {})

        # Check if the data is empty
        if not time_series:
            raise ValueError(f"No data found for symbol {self.stock_symbol} with interval {self.interval}")

        # Transform the data into a DataFrame
        self.df = pd.DataFrame.from_dict(time_series, orient='index')
        self.df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Convert index to datetime and filter data from the last 6 hours
        self.df.index = pd.to_datetime(self.df.index)
        three_hours_ago = datetime.now() - timedelta(hours=6)
        self.df = self.df[self.df.index >= three_hours_ago]

        # If the dataframe is still empty, extend the timeframe
        if self.df.empty:
            print(f"Not enough data for the last 6 hours for {self.stock_symbol}. Extending timeframe to 1 day.")
            one_day_ago = datetime.now() - timedelta(days=1)
            self.df = pd.DataFrame.from_dict(time_series, orient='index')
            self.df.index = pd.to_datetime(self.df.index)
            self.df = self.df[self.df.index >= one_day_ago]

        # Sort it to plot chronologically
        self.df = self.df.sort_index()

        # Convert necessary columns to numeric type
        self.df['Open'] = pd.to_numeric(self.df['Open'], errors='coerce')
        self.df['High'] = pd.to_numeric(self.df['High'], errors='coerce')
        self.df['Low'] = pd.to_numeric(self.df['Low'], errors='coerce')
        self.df['Close'] = pd.to_numeric(self.df['Close'], errors='coerce')
        self.df['Volume'] = pd.to_numeric(self.df['Volume'], errors='coerce')

        # Drop rows with NaN values
        self.df.dropna(inplace=True)

        # Resample data to ensure we have consistent intervals (if required)
        if not self.df.empty:
            self.df = self.df.resample('5T').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

    def plot_and_save_chart(self):
        # Plot and save the candlestick chart
        if self.df is None or self.df.empty:
            raise ValueError("Dataframe is empty. Please fetch data before plotting.")

        global SAVE_FOLDER_NAME
        # Create a new folder with the current date and time (DDMMYYHHSS) under the save location
        SAVE_FOLDER_NAME = datetime.now().strftime("%d%m%y%H%S")
        save_folder = os.path.join(self.save_location, SAVE_FOLDER_NAME)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Generate dynamic filename based on stock symbol and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.stock_symbol}_{timestamp}.png"
        save_path = os.path.join(save_folder, filename)

        # Plotting the candlestick chart and saving it to a file
        try:
            mpf.plot(self.df, type='candle', style='charles', title=f'{self.stock_symbol} {self.interval} Candlestick Chart (Last 3 Hours)',
                     ylabel='Price (USD)', volume=True, ylabel_lower='Volume', savefig=save_path)
            print(f"Chart saved successfully to {save_path}")
        except Exception as e:
            raise ValueError(f"Failed to plot and save chart for {self.stock_symbol}: {e}")

        return save_path

class PyBlogger:
    def __init__(self, directory, logo_path):
        # Initialize with directory to save blog content and logo path
        self.directory = directory.strip()
        self.logo_path = logo_path
        self.blog_content = ""
        self.chart_paths = []

    def add_chart_path(self, chart_path):
        # Add chart path to the list
        self.chart_paths.append(chart_path)

    def extract_images(self):
        # Extract stock images from the given directory
        self.image_paths = [
            os.path.join(self.directory, file) for file in os.listdir(self.directory)
            if file.endswith('.png')
        ]

    def analyze_image_chart(self, image_path, groq_key):
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Call LLM API for analysis
        image_query = "Analyze the stock chart and provide insights on significant trends and key indicators."
        max_tokens = 500
        temperature = 0.5
        top_p = 1.0
        analysis = self.call_llm_api(groq_key, base64_image, image_query, max_tokens, temperature, top_p)
        print(f"Chart Analysis for {image_path}: {analysis}")  # Verbose print statement for analysis
        return analysis

    def call_llm_api(self, groq_key, base64_image, image_query, max_tokens, temperature, top_p):
        # Call Groq LLM API to get insights from the image
        try:
            client = Groq(api_key=groq_key)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": image_query},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model="llama-3.2-11b-vision-preview",
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False,
                stop=None,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Failed to get response from LLM API: {str(e)}")
            return ""

    def upload_image_to_medium(self, medium_key, image_path):

        # Replace with your public and private keys
        public_key = UPLOAD_CARE_PUBLIC_KEY
        secret_key = UPLOAD_CARE_PRIVATE_KEY


        uploadcare = Uploadcare(public_key, secret_key)
        with open(image_path, "rb") as file_object:
            cdn_url = uploadcare.upload(file_object)


        print(f"File uploaded successfully. CDN URL: {cdn_url}")

        return cdn_url


    def write_blog(self, stock_symbols, groq_key, medium_key):
        # Fetch news for each stock symbol for the last 3 hours
        current_time = datetime.now()
        three_hours_ago = current_time - timedelta(hours=3)
        _from = three_hours_ago.strftime('%Y-%m-%d')
        _to = current_time.strftime('%Y-%m-%d')

        # Initialize Finnhub client
        finnhub_client = finnhub.Client(api_key=finnhub_token)

        for symbol, keyword in stock_symbols.items():
            try:
                news = finnhub_client.company_news(symbol, _from=_from, to=_to)
                # Filter news that contains the keyword in the headline
                news = [article for article in news if keyword.lower() in article['headline'].lower()]
            except Exception as e:
                print(f"Failed to fetch news or no relevant news found for symbol {symbol}. Error: {e}")
                continue

            if news:
                self.blog_content += f"# Stock Symbol: {symbol}\n\n"

            # Add chart image to the blog content using stored chart paths
            chart_path = next((path for path in self.chart_paths if symbol in path), None)
            if chart_path:
                medium_image_url = self.upload_image_to_medium(medium_key, chart_path)
                if medium_image_url:
                    self.blog_content += f"![Chart for {symbol}]({medium_image_url})\n\n"
                # Analyze the chart and add analysis to the blog content
                chart_analysis = self.analyze_image_chart(chart_path, groq_key)
                self.blog_content += f"**Chart Analysis**: {chart_analysis}\n\n"

            # Display news in bullet points and prepare data for sentiment analysis
            for article in news[:5]:
                headline = article['headline']
                summary = article.get('summary', 'No summary available')
                published_date = article.get('datetime')
                if published_date:
                    published_date_str = datetime.utcfromtimestamp(published_date / 1000).isoformat()
                else:
                    published_date_str = 'N/A'
                self.blog_content += f"- **Headline**: {headline}\n"
                self.blog_content += f"- **Summary**: {summary}\n"

                # Prepare Groq chat completion request
                try:
                    client = Groq(api_key=groq_key)
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": f"Based on the Stock Headline: '{headline}' and Summary: '{summary}', provide a one-word recommendation: BUY, SELL, or HOLD, and briefly explain why."
                            }
                        ],
                        model="llama3-8b-8192",
                    )
                    response_content = chat_completion.choices[0].message.content.strip()
                    sentiment_analysis = response_content.split('.')[0].upper()
                    reason = response_content
                    if sentiment_analysis not in ['BUY', 'SELL', 'HOLD']:
                        sentiment_analysis = 'HOLD'
                    print(f"Sentiment Analysis for {headline}: {sentiment_analysis} - {reason}")  # Verbose print statement for sentiment analysis
                except Exception as e:
                    print(f"Failed to get sentiment analysis for symbol {symbol}. Error: {e}")
                    sentiment_analysis = "Sentiment analysis not available."
                    reason = ""

                # Add sentiment analysis to blog content
                self.blog_content += f"- **Sentiment Analysis**: {sentiment_analysis}\n"
                self.blog_content += f"- **Reason**: {reason.strip()}\n"
                self.blog_content += f"- *Note: This analysis is based on an AI model recommendation and is not meant to be financial advice.*\n\n"

        # Add logo to the end of the blog content
        if self.logo_path:
            medium_logo_url = self.upload_image_to_medium(medium_key, self.logo_path)
            if medium_logo_url:
                self.blog_content += f"![logo displayed]({medium_logo_url})\n\n"

        # Save blog content to a file
        blog_file = os.path.join(self.directory, f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(blog_file, 'w') as file:
            file.write(self.blog_content)
        print(f"Blog content saved successfully to {blog_file}")

        # Post to Medium (example implementation)
        user_confirmation = input("Do you want to publish the blog to Medium? (yes/no): ").strip().lower()
        if user_confirmation == 'yes':
            self.post_to_medium(medium_key, blog_file)
        else:
            print(f"Blog content saved locally at {blog_file}, but not published to Medium.")

    def post_to_medium(self, medium_key, blog_file):

        print('post_to_medium called ')
        # Create blog post title with current timestamp
        blog_title = f"Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Read the blog content from the saved markdown file
        with open(blog_file, 'r') as file:
            blog_content = file.read()

        # Update image paths in the blog content
     #   updated_blog_content = ""
     #   lines = blog_content.splitlines()
      #  for line in lines:
      #      print(line)
      #      if line.startswith("![Chart for") or line.startswith("![logo displayed"):
      #          # Extract image path
      #          start_index = line.find('(') + 1
      #          end_index = line.find(')')
      #          image_path = line[start_index:end_index]
      #          # Upload image to Medium and get the URL
      #          medium_image_url = self.upload_image_to_medium(medium_key, image_path)
      #          print(image_path)
      #          print(medium_image_url)
      #          if medium_image_url:
       #             updated_blog_content += line.replace(image_path, medium_image_url) + "\n"
       #         else:
       #             updated_blog_content += line + "\n"
       #     else:
       #         updated_blog_content += line + "\n"
        updated_blog_content = blog_content
        print(updated_blog_content)
        # Fetch Medium user ID dynamically
        headers = {
            'Authorization': f'Bearer {medium_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'host': 'api.medium.com',
            'Accept-Charset': 'utf-8'
        }

        try:
            user_response = requests.get("https://api.medium.com/v1/me", headers=headers, timeout=10)
            user_response.raise_for_status()
            user_data = user_response.json()
            USER_ID = user_data['data']['id']
        except RequestException as e:
            print(f"Failed to retrieve user ID from Medium. Error: {e}")
            exit()

        # Prepare Medium API request
        url = f'https://api.medium.com/v1/users/{USER_ID}/posts'

        headers = {
            "Authorization": f"Bearer {medium_key}",
            "Content-Type": "application/json",
        }

        data = {
            "title": blog_title,
            "contentFormat": "markdown",
            "content": updated_blog_content,
            "tags": ["stocks", "AI analysis", "finance"],
            "publishStatus": "public"
        }

        # Post to Medium
        try:
            response = requests.post(url=url, headers=headers, json=data)
            response.raise_for_status()
            print('Blog post created successfully on Medium!')
        except RequestException as e:
            print(f"Failed to create blog post. Error: {e}")

# Example usage
if __name__ == "__main__":
    print("Welcome to StockInsighter!")
    time.sleep(1)

    # Ask user if they want to use a config file
    use_config = input("Do you want to use a configuration file for input values? (yes/no): ").strip().lower()
    if use_config == 'yes':
        config_file_path = input("Enter the config file path: ").strip()
        try:
            with open(config_file_path, 'r') as config_file:
                config_lines = config_file.readlines()
                config_values = {line.split(': ')[0].strip(): line.split(': ')[1].strip() for line in config_lines}
                api_key = config_values.get("Enter your Alpha Vantage API key")
                interval = config_values.get("Enter the interval (e.g., 5min, 15min, 30min)")
                save_location = config_values.get("Enter the location to save the charts")
                groq_key = config_values.get("Enter your Groq API key")
                medium_key = config_values.get("Enter your Medium API key")
                finnhub_token = config_values.get("Enter your Finnhub API key")
                logo_path = config_values.get("Enter your logo path")
                UPLOAD_CARE_PUBLIC_KEY = config_values.get("Enter UPLOAD_CARE_PUBLIC_KEY")
                UPLOAD_CARE_PRIVATE_KEY = config_values.get("Enter UPLOAD_CARE_PRIVATE_KEY")
                stock_symbols_input = config_values.get("Stock Symbols").split(',')
                stock_symbols = [symbol.split('-')[0].strip().upper() for symbol in stock_symbols_input]
                stock_keywords = {symbol.split('-')[0].strip().upper(): symbol.split('-')[1].strip() for symbol in stock_symbols_input}

        except Exception as e:
            print(f"Failed to read config file. Error: {e}")
            sys.exit(1)
    else:
        # User inputs for stock symbols, API keys, and intervals
        stock_symbols_input = input("Enter a list of stock symbols along with keywords (e.g., PLTR-Palantir) separated by commas: ").split(',')
        stock_symbols = [symbol.split('-')[0].strip().upper() for symbol in stock_symbols_input]
        stock_keywords = {symbol.split('-')[0].strip().upper(): symbol.split('-')[1].strip() for symbol in stock_symbols_input}
        api_key = input("Enter your Alpha Vantage API key: ")
        interval = input("Enter the interval (e.g., 5min, 15min, 30min): ")
        save_location = input("Enter the location to save the charts: ").strip()
        groq_key = input("Enter your Groq API key: ")
        medium_key = input("Enter your Medium API key: ")
        finnhub_token = input("Enter your Finnhub API key: ")
        logo_path = input("Enter the location of your logo: ")
        UPLOAD_CARE_PUBLIC_KEY = input("Enter UPLOAD_CARE_PUBLIC_KEY: ")
        UPLOAD_CARE_PRIVATE_KEY = input("Enter UPLOAD_CARE_PRIVATE_KEY: ")

   # Create PyBlogger instance
    py_blogger = PyBlogger(save_location, logo_path)

    # Loop through each stock symbol and generate the chart
    for stock_symbol in stock_symbols:
        stock_chart = StockCandlestickChart(api_key, stock_symbol, interval, save_location)
        try:
            time.sleep(12)  # Adding delay to avoid hitting API limits
            stock_chart.fetch_data()
            chart_path = stock_chart.plot_and_save_chart()
            py_blogger.add_chart_path(chart_path)
        except Exception as e:
            print(f"Failed to generate chart for {stock_symbol}: {e}")

    print("\nPhase 1 completed successfully!")
    time.sleep(1)

    # Phase 2: Blog creation
    proceed_phase_2 = input("\nDo you want to proceed to Phase 2 (News Fetch and Blog Creator)? (yes/no): ").strip().lower()
    if proceed_phase_2 == 'yes':
        print("\nStarting Phase 2: News Fetch and Blog Creator...")
        py_blogger.write_blog(stock_keywords, groq_key, medium_key)
    else:
        print("\nThank you for using StockInsighter! Goodbye!")
