import streamlit as st
import os
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai.tools import tool
from dotenv import load_dotenv

# 1. Page Config
st.set_page_config(page_title="AI Financial Analyst", page_icon="📈")

# 2. Header
st.title("🤖 AI Financial Analyst Agent")
st.markdown("Enter a stock ticker (e.g., **TATASTEEL.NS**, **RELIANCE.NS**, **AAPL**) and let the AI Crew analyze it for you.")

# 3. Sidebar for API Key
with st.sidebar:
    st.header("⚙️ Configuration")
    load_dotenv()
    default_key = os.getenv("GOOGLE_API_KEY") if os.getenv("GOOGLE_API_KEY") else ""
    api_key = st.text_input("Enter Google API Key", value=default_key, type="password")

# 4. Input Box
stock_ticker = st.text_input("Enter Stock Ticker:", value="TATASTEEL.NS")

# --- TOOL DEFINITION (The Fix: Native CrewAI Way) ---
class StockTools:
    @tool("Get Stock Price")
    def fetch_stock_price(ticker: str):
        """
        Useful for getting the live price of a stock. 
        Input should be a ticker symbol like 'TATASTEEL.NS' or 'RELIANCE.NS'.
        """
        try:
            ticker = ticker.strip()
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            if price:
                return f"The live price of {ticker} is ₹{price}"
            else:
                return f"Error: Could not find price for {ticker}. Try adding '.NS' for Indian stocks."
        except Exception as e:
            return f"Error fetching price: {e}"

# --- MAIN LOGIC ---
if st.button("🚀 Analyze Stock"):
    if not api_key:
        st.error("Please enter your Google API Key in the sidebar!")
    else:
        with st.spinner('🤖 AI Crew is researching the market... (This may take 30-60 seconds)'):
            try:
                # Setup Brain
                llm = ChatGoogleGenerativeAI(
                    model="gemini-flash-latest",
                    verbose=True,
                    temperature=0.5,
                    google_api_key=api_key
                )

                # Define Agents
                researcher = Agent(
                    role='Senior Stock Market Researcher',
                    goal='Find live stock prices and analyze trends',
                    backstory="You are an expert analyst. You ALWAYS use the tool to find real data.",
                    verbose=True,
                    llm=llm,
                    # 👇 FIX: Correct way to pass the tool
                    tools=[StockTools.fetch_stock_price]
                )

                writer = Agent(
                    role='Financial Blog Writer',
                    goal='Write a blog post with the LIVE PRICE included',
                    backstory="You write engaging blogs in Hinglish. You must mention the exact price found by the researcher.",
                    verbose=True,
                    llm=llm
                )

                # Define Tasks
                task1 = Task(
                    description=f"""
                    1. Use the 'Get Stock Price' tool to find the LIVE price of '{stock_ticker}'.
                    2. Analyze if it is a good time to buy based on general market trends.
                    """,
                    expected_output="A summary report with the exact LIVE price.",
                    agent=researcher
                )

                task2 = Task(
                    description=f"""
                    Write a short Hinglish blog post titled '{stock_ticker} Update'.
                    IMPORTANT: You MUST include the exact Live Price (₹) found by the researcher.
                    Format the output nicely with Markdown (Headings, Bold text).
                    """,
                    expected_output="A blog post with the Real-Time Price mentioned.",
                    agent=writer
                )

                # Create Crew
                my_crew = Crew(
                    agents=[researcher, writer],
                    tasks=[task1, task2],
                    process=Process.sequential
                )

                # Run!
                result = my_crew.kickoff()

                # Display Result
                st.success("Analysis Complete! ✅")
                st.markdown("---")
                st.markdown(result)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")