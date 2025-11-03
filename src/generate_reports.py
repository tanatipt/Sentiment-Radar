import os
import time
import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict
from dotenv import load_dotenv
from src.graph_constructor import GraphConstructor
from config import settings
from typing_extensions import Literal

logging.basicConfig(level=logging.INFO)
load_dotenv()

def send_email(subject: str, body: str, sender: str, recipient: str, password: str) -> None:
    """Send an email with the given subject and body.

    Args:
        subject (str): Email subject
        body (str): Body of the email
        sender (str): Email address of sender
        recipient (str): Email address of recipient
        password (str): Sender's email password
    Returns:
        None
    """
    msg = MIMEText(body, 'html')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(sender, password)
            smtp_server.sendmail(sender, recipient, msg.as_string())
        logging.info(f"Email sent: {subject}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

def format_sections(sections: List[str]) -> str:
    """
    Formats a list of section strings into an HTML document.

    Args:
        sections (List): A list of strings, each representing a section to be included in the HTML body.

    Returns:
        str: A string containing the formatted HTML document with the sections joined by double newlines.
    """
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <body>
        {'<br><br>'.join(sections)}
    </body>
    </html>
    """

def generate_report_for_symbol(
        asset_type: Literal["cryptocurrency", "stocks"], 
        symbol: str, 
        exchange: Literal["BINANCE", "NASDAQ"], 
        alias: str
) -> str:
    """
    Generate the sentiment report for a given trading asset

    Args:
        asset_type (Literal[cryptocurrency, stocks]): Asset type
        symbol (str): Trading symbol of asset
        exchange (Literal[BINANCE, NASDAQ]): Exchange where asset is traded
        alias (str): Alias for the trading asset

    Returns:
        str: Email of the sentiment report
    """
    try:
        graph = GraphConstructor(
            generator_config = settings.generator,
            critic_config = settings.critic,
            asset_information = {
                "asset_type" : asset_type,
                "trading_symbol": symbol,
                "trading_exchange": exchange,
                "symbol_alias" : alias
            }
        ).compile()

        response = graph.invoke(input={}, config={"recursion_limit": 100})
        email = response.get("email")

        if email is None:
            logging.error(f"Report generation failed for {symbol}")
            return ""
        
        logging.info(f"Report generated for {symbol}")
        return email.strip("`").removeprefix("html\n")
    except Exception as e:
        logging.error(f"Error generating report for {symbol}: {e}")
        return ""
    finally:
        time.sleep(30)  # Avoid API rate limits

def generate_and_send_reports(exchanges: Dict[str, str]) -> None:
    """

    Generate and email reports for all exchanges.

    Args:
        exchanges (Dict[str, str]): A dictionary of exchanges and asset types.
    Returns:
        None
    """
    sender = os.getenv("GMAIL_ADDRESS")
    recipient = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_PASSWORD")
    current_day = datetime.now(ZoneInfo('Asia/Bangkok')).strftime('%Y-%m-%d')

    for exchange, asset_type in exchanges.items():
        subject = f"{current_day} {asset_type.capitalize()} Sentiment Report"
        trading_symbols = settings.assets.get(exchange, {})
        sections = []

        for alias, symbol in trading_symbols.items():
            section = generate_report_for_symbol(asset_type, symbol, exchange, alias)
            if section:
                sections.append(section)

        if sections:
            html_content = format_sections(sections)
            send_email(subject, html_content, sender, recipient, password)

if __name__ == "__main__":
    # Map exchanges to their asset types
    EXCHANGES = {
        "BINANCE": "cryptocurrency",
        "NASDAQ": "stocks"
    }
    generate_and_send_reports(EXCHANGES)
