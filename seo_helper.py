# import streamlit as st
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin
# import pandas as pd

# # Function to fetch and parse a page
# def fetch_page(url):
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, 'html.parser')
#         return soup
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error fetching {url}: {e}")
#         return None

# # Function to extract links from a page
# def extract_links(base_url, soup):
#     links = set()
#     for a_tag in soup.find_all('a', href=True):
#         href = a_tag['href']
#         full_url = urljoin(base_url, href)
#         if base_url in full_url:  # Only keep internal links
#             links.add(full_url.split('#')[0])  # Remove fragment identifiers
#     return links

# # Expanded metadata extraction
# def extract_metadata(url, soup):
#     title = soup.title.string if soup.title else "No Title"
    
#     # Meta description
#     meta_desc = soup.find('meta', attrs={'name': 'description'})
#     description = meta_desc['content'] if meta_desc else "No Meta Description"
    
#     # Canonical tag
#     canonical_tag = soup.find('link', rel='canonical')
#     canonical = canonical_tag['href'] if canonical_tag else "No Canonical URL"
    
#     # Headers (H1, H2)
#     h1 = [h1_tag.get_text(strip=True) for h1_tag in soup.find_all('h1')]
#     h2 = [h2_tag.get_text(strip=True) for h2_tag in soup.find_all('h2')]
    
#     return {
#         "URL": url,
#         "Title": title,
#         "Meta Description": description,
#         "Canonical URL": canonical,
#         "H1 Tags": ", ".join(h1) if h1 else "No H1 Tags",
#         "H2 Tags": ", ".join(h2) if h2 else "No H2 Tags"
#     }

# # Recursive function to crawl pages
# def crawl_website(base_url, max_pages=10):
#     to_crawl = set([base_url])
#     crawled = set()
#     results = []

#     while to_crawl and len(crawled) < max_pages:
#         url = to_crawl.pop()
#         if url in crawled:
#             continue
        
#         soup = fetch_page(url)
#         if soup:
#             # Extract expanded metadata and links
#             metadata = extract_metadata(url, soup)
#             results.append(metadata)
#             new_links = extract_links(base_url, soup)
#             to_crawl.update(new_links - crawled)

#         crawled.add(url)
    
#     return results

# # Streamlit UI
# st.title("Website Audit Tool - Expanded Metadata Extraction")

# base_url = st.text_input("Enter the website URL:", "https://example.com")
# max_pages = st.slider("Max pages to crawl:", 1, 50, 10)

# if st.button("Start Crawl"):
#     if base_url:
#         st.info(f"Crawling {base_url}...")
#         crawl_results = crawl_website(base_url, max_pages)
        
#         if crawl_results:
#             df = pd.DataFrame(crawl_results)
#             st.write("### Crawl Results", df)
#             st.download_button("Download CSV", df.to_csv(index=False), file_name="expanded_crawl_results.csv")
#         else:
#             st.warning("No data found. Please check the URL or try a different site.")








### Old Code ####




import streamlit as st
from urllib.parse import unquote
import gsc_data_pull 
import requests
from bs4 import BeautifulSoup
from llm_integration import query_gpt 

# Page configuration
st.set_page_config(layout="wide")

def fetch_page_copy(url):
    try:
        # Fetch the content of the page
        response = requests.get(url)
        response.raise_for_status()  # Check if request was successful

        # Parse the page content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the title tag
        title = soup.title.string if soup.title else "No title found"

        # Extract the meta description
        meta_description = ""
        description_tag = soup.find("meta", attrs={"name": "description"})
        if description_tag and description_tag.get("content"):
            meta_description = description_tag["content"]
        else:
            meta_description = "No meta description found"

        # Extract meta keywords
        meta_keywords = ""
        keywords_tag = soup.find("meta", attrs={"name": "keywords"})
        if keywords_tag and keywords_tag.get("content"):
            meta_keywords = keywords_tag["content"]
        else:
            meta_keywords = "No meta keywords found"

        # Extract main text from <p> and heading tags
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3'])
        page_text = "\n\n".join([para.get_text(strip=True) for para in paragraphs])

        # Combine all extracted data into a dictionary
        seo_data = {
            "Title": title,
            "Meta Description": meta_description,
            "Meta Keywords": meta_keywords,
            "Page Copy": page_text if page_text else "No main content found on this page."
        }

        return seo_data
    except requests.RequestException as e:
        return {"Error": f"An error occurred while fetching the page: {e}"}

def display_report_with_llm(llm_prompt):
    # Query the LLM with the prompt
    llm_response = query_gpt(llm_prompt)
    st.write("GPT-4 Analysis:")
    st.write(llm_response)

def main():
    # Ensure session_summary is initialized in session state
    if "session_summary" not in st.session_state:
        st.session_state["session_summary"] = ""  # Initialize with an empty string or default value

    
    # Pull the same dataframe as in the main app
    df = gsc_data_pull.fetch_search_console_data()  # Replace 'pull_data' with the actual function name
    
    # Retrieve message from URL parameter
    query_params = st.experimental_get_query_params()
    message = query_params.get("message", ["No message received"])[0]

    # Display SEO helper app
    st.title("SEO Helper")
    st.write("This is the SEO helper app.")

    # Input field for the URL to scrape
    url = st.text_input("Enter a URL to scrape", placeholder="https://example.com")
    
    if url:
        st.write("Fetching content...")
        seo_data = fetch_page_copy(url)

        with st.expander("See Website Copy"):
            st.subheader("SEO Information")
            st.write(f"**Title:** {seo_data['Title']}")
            st.write(f"**Meta Description:** {seo_data['Meta Description']}")
            st.write(f"**Meta Keywords:** {seo_data['Meta Keywords']}")
            st.subheader("Page Copy")
            st.write(seo_data["Page Copy"])

        # Generate the prompt for LLM analysis
        llm_prompt = (
            f"Here is the SEO information and page copy from a webpage:\n\n"
            f"Title: {seo_data['Title']}\n"
            f"Meta Description: {seo_data['Meta Description']}\n"
            f"Meta Keywords: {seo_data['Meta Keywords']}\n"
            f"Page Copy: {seo_data['Page Copy']}\n\n"
            f"Based on this SEO information, please suggest possible improvements. Have one section main section that talks about overall SEO strategy. Below that have another section where you identify actual pieces of text you see that could be tweaked."
            f"Use the following context to guide your suggestions: {message}. "
            f"This is an analysis from an initial look at the search query report from this website."
        )

        # Display LLM analysis
        display_report_with_llm(llm_prompt)
 
if __name__ == "__main__":
     main()
