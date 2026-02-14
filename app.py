import streamlit as st
import os
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai.tools import tool

# --- 🚨 HARDCODED API KEY (Sirf Testing Ke Liye) ---
MY_API_KEY = "AIzaSyAhcoN_nMYmg6VVIYV9pF4XVRmbiiMzuZ4"

# Naye version ke liye hum dono environment variables set kar rahe hain
os.environ["GOOGLE_API_KEY"] = MY_API_KEY
os.environ["GEMINI_API_KEY"] = MY_API_KEY 

# 1. Page Config
st.set_page_config(page_title="AI Financial Analyst", page_icon="📈")

# 2. Header
st.title("🤖 AI Financial Analyst Agent")
st.markdown("Enter a stock ticker (e.g., **TATASTEEL.NS**, **RELIANCE.NS**) and let the AI Crew analyze it.")

# 3. Input Box
stock_ticker = st.text_input("Enter Stock Ticker:", value="TATASTEEL.NS")

# --- TOOL DEFINITION ---
@tool
def get_stock_price(ticker: str):
    """Useful to get the live stock price. Input: Ticker symbol (e.g., 'TATASTEEL.NS')."""
    try:
        ticker = ticker.strip()
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        if price:
            return f"The live price of {ticker} is ₹{price}"
        else:
            return f"Error: Price not found for {ticker}."
    except Exception as e:
        return f"Error: {e}"

# --- MAIN LOGIC ---
if st.button("🚀 Analyze Stock"):
    with st.spinner('🤖 AI Crew is researching... (Please wait 30-40 seconds)'):
        try:
            # Setup Brain - Error ke hisaab se humne seedha 'api_key' parameter use kiya hai
            llm = ChatGoogleGenerativeAI(
                model="gemini-flash-latest",
                verbose=True,
                temperature=0.5,
                api_key=MY_API_KEY 
            )

            # Agents
            researcher = Agent(
                role='Senior Stock Market Researcher',
                goal='Find live stock prices',
                backstory="Expert analyst who uses tools to find data.",
                verbose=True,
                llm=llm,
                tools=[get_stock_price]
            )

            writer = Agent(
                role='Financial Blog Writer',
                goal='Write a blog with LIVE PRICE',
                backstory="Writes engaging blogs in Hinglish.",
                verbose=True,
                llm=llm
            )

            # Tasks
            task1 = Task(
                description=f"Find the LIVE price of '{stock_ticker}' using the tool.",
                expected_output="A report with the exact LIVE price.",
                agent=researcher
            )

            task2 = Task(
                description=f"Write a short Hinglish blog about '{stock_ticker}'. INCLUDE THE LIVE PRICE.",
                expected_output="A blog post with the price.",
                agent=writer
            )

            # Crew
            my_crew = Crew(
                agents=[researcher, writer],
                tasks=[task1, task2],
                process=Process.sequential
            )

            result = my_crew.kickoff()

            st.success("Analysis Complete! ✅")
            st.markdown(result)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
