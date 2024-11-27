import openai
import streamlit as st
import pandas as pd
from datetime import date, timedelta
from ga4_data_pull import *
from gsc_data_pull import *
from llm_integration import *
from urllib.parse import quote

# Page configuration
st.set_page_config(page_title="BizBuddy", layout="wide", page_icon = "🤓")

st.markdown("<h1 style='text-align: center;'>Welcome to BizBuddy: Let's Grow Your Digital Presence</h1>", unsafe_allow_html=True)

def generate_seo_insights(search_data):
   # Prepare the search query list
   query_list = search_data["Search Query"].unique()
   formatted_queries = "\n".join(query_list)

   # Define the prompt for the LLM
   prompt = (
   "Here are the search queries this website currently appears for:\n"
   f"{formatted_queries}\n\n"
   "Based on this data, please provide the following, make sure to bold any suggested keywords:\n"
   "- Target search terms that align with the website's goals.\n"
   "- New niche ideas for search terms that could improve conversions.\n"
   "- A brief explanation of why SEO optimization is critical for this business."
   )

   # Call the LLM using query_gpt
   response = query_gpt(prompt)
   return response
   
# Initialize LLM context with business context on app load
initialize_llm_context()

# Generate and display each summary with LLM analysis
def display_report_with_llm(summary_func, llm_prompt):
   # Generate summary
   summary = summary_func()
   
   # Query LLM with specific prompt
   llm_response = query_gpt(llm_prompt, summary)
   return llm_response


def main():
    # Fetch data for the last 30 days (from 30 days ago to yesterday)
    start_date_30_days = "30daysAgo"
    end_date_yesterday = "yesterday"

    df_30_days = fetch_metrics_by_source(start_date_30_days, end_date_yesterday)
    
    # Fetch data for the last month (from 60 days ago to 30 days ago)
    start_date_60_days = "60daysAgo"
    end_date_30_days = "31daysAgo"

    df_60_to_30_days = fetch_metrics_by_source(start_date_60_days, end_date_30_days)

    # Fetch event data (generate leads)
    event_data = fetch_metrics_by_event(start_date_30_days, end_date_yesterday)  # Add this line to fetch event data

    lp_df_30_days = fetch_metrics_by_landing_page(start_date_30_days, end_date_yesterday)
   
    # First column - GA4 Metrics and Insights
    col1, col2 = st.columns(2)
   
    with col1:
        st.markdown("<h3 style='text-align: center;'>Web Performance Overview</h3>", unsafe_allow_html=True)

        # Fetch event data (generate leads)
        last_month_event_data = fetch_metrics_by_event(start_date_60_days, start_date_30_days)  # Add this line to fetch event data
        
        # Summarize monthly data with leads now included (for the 30 days data)
        current_summary = summarize_monthly_data(df_30_days, event_data)[0]
        last_month_summary = summarize_last_month_data(df_60_to_30_days, last_month_event_data)[0]
       
        # Display GA4 metrics (Updated with the new leads data)
        generate_all_metrics_copy(current_summary, last_month_summary)
        
        # LLM insights based on GA data
        ga_llm_prompt = """
           Based on the following website performance metrics, provide a short analysis. Highlight key improvements, areas needing attention, 
           and how these metrics compare to typical industry standards. Limit your response to 2-3 bullet points.
           """
        
        # Combine current summary into a string for LLM processing
        metric_summary_text = "\n".join([f"{row['Metric']}: {row['Value']}" for _, row in current_summary.iterrows()])
        ga_insights = query_gpt(ga_llm_prompt, metric_summary_text)
        
        st.markdown("### Insights from AI")
        st.markdown(ga_insights)

    # Second column - Acquisition Overview (with Pie Chart and Source Descriptions)
    with col2:
        st.markdown("<h3 style='text-align: center;'>Acquisition Overview</h3>", unsafe_allow_html=True)
        acq_col1, acq_col2 = st.columns(2)
    with acq_col1:
        plot_acquisition_pie_chart_plotly(summarize_monthly_data(df_30_days, event_data)[1])
    with acq_col2:
        describe_top_sources(summarize_monthly_data(df_30_days, event_data)[1])
        
        temp_url = "https://bizbuddyv1-ppcbuddy.streamlit.app/"
        st.markdown("Search and social ads are key to driving traffic. Check out these tools to help you get going.")
        st.link_button("Paid Search - Helper", temp_url)
        st.link_button("Social Ads - Helper", temp_url)

    # Landing page analysis section
    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<h3 style='text-align: center;'>Individual Page Overview</h3>", unsafe_allow_html=True)
    
        # Get landing page summary (now includes leads)
        landing_page_summary = summarize_landing_pages(lp_df_30_days, event_data)
        generate_page_summary(landing_page_summary)
        
        llm_input = st.session_state.get("page_summary_llm", "")
        response = query_gpt("Provide insights based on the following page performance data, note that there is no CTAs on any page besides the Home. We need to think of ways to drive more people to the contact page. State only the bullets, no pre text. Limit your response to 2-3 bullet points:", llm_input)
        
        st.markdown("### Insights from AI")
        st.markdown(response)
    
    with col4:
        st.markdown("<h3 style='text-align: center;'>Search Query Analysis</h3>", unsafe_allow_html=True)
        sq_col1, sq_col2 = st.columns(2)
    with sq_col1:
        st.markdown("These are all the search terms that your website has shown up for in the search results. The Google search engine shows websites based on the relevance of a website's information as it relates to the search terms.")
        search_data = fetch_search_console_data()
        st.dataframe(search_data['Search Query'], use_container_width=True)
        
    with sq_col2:
        seo_insights = generate_seo_insights(search_data)
        st.markdown(seo_insights)
        encoded_message = quote(str(seo_insights))
        seo_url = f"https://bizbuddyv1-seobuddy.streamlit.app?message={encoded_message}"
        st.link_button("Check Out our SEO Helper!!", seo_url)

# Execute the main function only when the script is run directly
if __name__ == "__main__":
    main()
