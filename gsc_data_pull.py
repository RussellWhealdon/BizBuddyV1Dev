import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from google.oauth2 import service_account
import streamlit as st

# Define the Google Search Console property URL
PROPERTY_URL = 'https://www.chelseawnutrition.com/'  # Replace with your actual website URL in Search Console

# Load the service account credentials from Streamlit secrets
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["google_service_account"],
    scopes=['https://www.googleapis.com/auth/webmasters.readonly']
)

# Initialize the Google Search Console service
service = build('searchconsole', 'v1', credentials=credentials)

# Define a function to fetch Google Search Console data
def fetch_search_console_data(start_date=None, end_date=None):
    # Default to last 30 days if no date range is provided
    if not start_date:
        end_date = datetime.today()
        start_date = "2024-01-01"
    
    # Format dates as strings for the API
    start_date_str = start_date
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Create the request payload
    request = {
        'startDate': start_date_str,
        'endDate': end_date_str,
        'dimensions': ['query'],  # Break down by search query
        'searchType': 'web',
        'rowLimit': 1000  # Adjust if needed, max 25,000 per call
    }
    
    # Run the query
    response = service.searchanalytics().query(siteUrl=PROPERTY_URL, body=request).execute()
    
    # Parse response into a list of rows
    rows = []
    for row in response.get('rows', []):
        query = row['keys'][0]
        impressions = row.get('impressions', 0)
        clicks = row.get('clicks', 0)
        ctr = row.get('ctr', 0)
        position = row.get('position', 0)
        rows.append([query, impressions, clicks, ctr, position])
    
    # Load the data into a DataFrame
    df = pd.DataFrame(rows, columns=['Search Query', 'Impressions', 'Clicks', 'CTR', 'Avg. Position'])
    return df


# Function to create a summary of the top 30 search queries for LLM consumption
def summarize_search_queries(search_data):
    # Ensure necessary columns are present
    if not all(col in search_data.columns for col in ["Search Query", "Impressions", "Clicks", "Avg. Position"]):
        raise ValueError("Data does not contain required columns.")

    # Sort by Avg. Position and select top 30
    top_queries = search_data.sort_values(by="Avg. Position").head(30)
    top_queries["Avg. Position"] = top_queries["Avg. Position"].round(0).astype(int)
    
    # Format the summary as a readable text
    summary = "Top 30 Search Queries Summary:\n"
    summary += "Query | Impressions | Clicks | Avg. Position\n"
    summary += "-" * 50 + "\n"

    for _, row in top_queries.iterrows():
        query = row["Search Query"]
        impressions = row["Impressions"]
        clicks = row["Clicks"]
        avg_position = row["Avg. Position"]
        
        summary += f"{query} | {impressions} | {clicks} | {avg_position},\n"
    
    return summary
