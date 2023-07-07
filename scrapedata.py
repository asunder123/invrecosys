import requests
from bs4 import BeautifulSoup

def scrape_market_data(website):
    """Scrape market data from the given website."""
    response = requests.get(website)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the data from the website.
    data = soup.find_all('div', class_='stock-data')

    # Return the data.
    return data

def augment_data(data):
    """Augment the given data with market data from nifty and bloomberg."""
    # Scrape market data from nifty.
    nifty_data = scrape_market_data('https://www.nseindia.com/live-market/dynaContent/live_market_watch.jsp?symbol=NIFTY')

    # Scrape market data from bloomberg.
    bloomberg_data = scrape_market_data('https://www.bloomberg.com/quote/^BSESN')

    # Merge the data from nifty, bloomberg, and the original data.
    data = data.append(nifty_data).append(bloomberg_data)

    # Return the augmented data.
    return data

# Scrape market data from nifty and bloomberg.
data = scrape_market_data('https://www.nseindia.com/live-market/dynaContent/live_market_watch.jsp?symbol=NIFTY')
data = scrape_market_data('https://www.bloomberg.com/quote/^BSESN')

# Augment the data with market data from nifty and bloomberg.
data = augment_data(data)

# Print the augmented data.
print(data)