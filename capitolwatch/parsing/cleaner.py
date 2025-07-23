import re
from bs4 import BeautifulSoup

def clean_html_string(html):
    """
    Cleans a raw HTML string by removing unwanted tags, 
    special characters, and normalizing the text.

    Args:
        html (str): Raw HTML content as a string.

    Returns:
        str: Cleaned and normalized plain text.
    """
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    

    # Remove unnecessary tags (script, style, header, footer, etc.)
    for tag in soup(["script", "style", "noscript", "footer", "header", "nav", "aside"]):
        tag.decompose()

    
    # Extract plain text from HTML (keeping line breaks between blocks)
    text = soup.get_text(separator="\n")


    
    # Replace multiple whitespace characters (spaces, tabs, newlines) with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove special characters except basic punctuation (.,:;!?€$-)
    text = re.sub(r"[^\w\s.,:;!?€$-]", "", text)    

    return text

def clean_html_file(filepath):
    """
    Loads an HTML file from disk, cleans its content, and returns plain text.

    Args:
        filepath (str or Path): Path to the HTML file.

    Returns:
        str: Cleaned and normalized plain text from the file.
    """
    # Open and read the file content as a string
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    # Clean the loaded HTML string using the previous function
    return clean_html_string(html)