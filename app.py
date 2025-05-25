import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import datetime
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,accuracy_score

india_stock_file = "Stock_name/India.csv"
canada_stock_file = "Stock_name/canada.csv"
us_stock_file = "Stock_name/US_STOCKS.csv"

# Suppress warnings
warnings.filterwarnings("ignore")
# Set default pandas display options
pd.set_option('display.max_columns', None)
# --- Streamlit App Configuration & Styling ---
hide_streamlit_style = """
    <style>
    # MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after {
        content:'Dhruv Bhatt'; 
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
    }
    </style>
"""

st.set_page_config(
    page_title="Stock_Prediction",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': 'https://www.extremelycoolapp.com/bug',
        'About': "# This is an *extremely* cool app!"
    }
)

# Apply custom CSS
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .title { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to download data
@st.cache_data
def load_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False
    )
    # 3. If your columns came in as a MultiIndex (e.g. level 0 = "TSLA", level 1 = field),
#    drop the top level so you only have the field names:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    return df

# Compute performance metrics
@st.cache_data
def compute_metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
    df = df.copy()
    df['Daily Return'] = df['Close'].pct_change()
    print("Calculating performance metrics...")
    df.dropna(inplace=True)
    cumulative_return = df['Close'].iloc[-1] / df['Close'].iloc[0] - 1
    print("Cumulative return calculated:", cumulative_return)
    if df['Daily Return'].std() == 0:
        return cumulative_return, np.nan, np.nan, np.nan  # Avoid division by zero
    print("Performance metrics calculated.")
    average_daily_return = df['Daily Return'].mean()
    print("Average daily return:", average_daily_return)
    annual_volatility = df['Daily Return'].std() * np.sqrt(252)
    print("Annualized volatility:", annual_volatility)
    sharpe_ratio = (average_daily_return / df['Daily Return'].std()) * np.sqrt(252)
    print("Sharpe ratio:", sharpe_ratio)
    return cumulative_return, average_daily_return, annual_volatility, sharpe_ratio

# Calculate RSI
@st.cache_data
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Preprocess data for LSTM
def preprocess_data(df: pd.DataFrame) -> tuple[np.ndarray, MinMaxScaler]:
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Create sequences for LSTM
def create_dataset(dataset: np.ndarray, time_step: int = 60) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

# Build LSTM model
@st.cache_resource
def build_model(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

@st.cache_data
def load_ticker_list(country: str) -> pd.DataFrame:
    if country.upper() == 'INDIA':
        df = pd.read_csv(india_stock_file, encoding='ISO-8859-1', header=None, skiprows=1)
    elif country.upper() == 'CANADA':
        df = pd.read_csv(canada_stock_file, encoding='ISO-8859-1', header=None, skiprows=1)
        df.columns = ['Name', 'Symbol']
    else:
        df = pd.read_csv(us_stock_file)
    df.columns = ['Name', 'Symbol']
    return df

@st.cache_data
def find_symbol(df: pd.DataFrame, name: str) -> str:
    row = df[df['Name'] == name]
    return row['Symbol'].values[0] if not row.empty else None

@st.cache_data
def display_stock_profile(symbol: str, name: str) -> None:
    stock_ticker = yf.Ticker(symbol)
    with st.expander(f"{name} Profile", expanded=True):
        info = stock_ticker.info
        profile_data = {
            "Name": info.get("shortName", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "Current Price": info.get("currentPrice", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "P/E": info.get("trailingPE", "N/A"),
            "Book Value": info.get("bookValue", "N/A"),
            "Price to Book": info.get("priceToBook", "N/A"),
            "Return on Equity": f"{info.get('returnOnEquity')*100:.2f}%" if info.get("returnOnEquity") is not None else "N/A",
            "Volume": info.get("volume", "N/A"),
            "Earnings Growth": f"{info.get('earningsGrowth')*100:.2f}%" if info.get("earningsGrowth") is not None else "N/A",
            "Revenue Growth": f"{info.get('revenueGrowth')*100:.2f}%" if info.get("revenueGrowth") is not None else "N/A"
        }
        # Display profile data in columns
        items = list(profile_data.items())
        # Display items in rows of 3 columns each
        for i in range(0, len(items), 3):
            cols = st.columns(3)
            for j, (label, value) in enumerate(items[i:i+3]):
                with cols[j]:
                    st.markdown(f"{label}: **{value}**")

# @st.cache_data
def get_analysis_options():
    time_step = st.sidebar.slider("Time Step (look-back window For forecast)", min_value=10, max_value=100, value=60)
    show_ma= st.sidebar.checkbox("Show Moving Averages", value=True)
    moving_average= None
    # Moving average section with sidebar controls
    if show_ma:
        # moving_average = st.sidebar.selectbox(
        #     "Moving Average Section",
        #     ["None", "10-day MA", "50-day MA", "Show Both (10 & 50 days)"]
        # )
        moving_average= st.sidebar.slider(
            "Moving Average Days",
            min_value=5,
            max_value=60,
            value=10,
            step=1
        )
    show_rsi = st.sidebar.checkbox("Show RSI Trend")
    rsi_days = None
    if show_rsi:
        rsi_days = st.sidebar.slider("RSI Days", min_value=5, max_value=30, value=14)
        # st.sidebar.markdown("---")
        # st.sidebar.write(f"RSI trend will be calculated over {rsi_days} days.")
    show_volatility = st.sidebar.checkbox("Show Volatility Trend")
    volatility_days = None
    if show_volatility:
        volatility_days = st.sidebar.slider("Volatility Days", min_value=5, max_value=60, value=30)
        # st.sidebar.markdown("---")
        # st.sidebar.write(f"Volatility will be calculated over {volatility_days} days.")
    forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=5)
    # default value is 5%
    accuracy_threshold = st.sidebar.slider(
        "Accuracy Threshold (%)",
        min_value=1,
        max_value=100,
        value=5,
        step=1
    )
    st.sidebar.markdown("---")
    run_analysis = st.sidebar.button("Run Analysis & Prediction")

    return time_step, show_ma, moving_average, show_rsi, rsi_days, show_volatility, volatility_days, forecast_days, run_analysis, accuracy_threshold


def show_train_results(model, history):
    """
    Display training results and metrics in an expander.
    """

    if history is not None:
        with st.expander("Training Metrics"):
                st.write("LSTM model training completed. Below are the training metrics:")
                st.write(f"Final Loss: {history.history['loss'][-1]:.4f}")
                st.write(f"Final Mean Absolute Error (MAE): {history.history['mean_absolute_error'][-1]:.4f}")
                # Display training loss and MAE
                st.write("Training Loss and Mean Absolute Error (MAE) over epochs:")
                fig, ax = plt.subplots()
                ax.plot(history.history['loss'], label='Loss')
                ax.plot(history.history['mean_absolute_error'], label='MAE')
                ax.set_title('Training Loss and MAE')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Value')
                ax.legend()
                st.pyplot(fig)
    if model is not None:
        with st.expander("LSTM Model Summary"):
            st.write("LSTM model architecture and summary:")
            model.summary(print_fn=lambda x: st.text(x))
    
    # # also show weights and biases
    # if model is not None:
    #     with st.expander("LSTM Model Weights and Biases"):
    #         for layer in model.layers:
    #             weights = layer.get_weights()
    #             if weights:
    #                 st.write(f"Layer: {layer.name}")
    #                 for i, weight in enumerate(weights):
    #                     st.write(f"  Weight {i}: {weight.shape}")
    #                     st.write(weight)
    #                 st.write("---")
    #                 import matplotlib.pyplot as plt

    if model is not None:
        with st.expander("LSTM Weights & Biases (Plotly)"):
            for layer in model.layers:
                weights = layer.get_weights()
                if not weights:
                    continue

                st.write(f"### Layer: {layer.name}")
                for idx, w in enumerate(weights):
                    title = f"{layer.name} – tensor #{idx}"

                    # 1) 2D weight matrices → heatmap
                    if w.ndim == 2:
                        fig = px.imshow(
                            w,
                            labels={'x':'Input units','y':'Output units','color':'Weight'},
                            title=title,
                            aspect='auto'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # 2) 1D bias vectors → bar chart
                    elif w.ndim == 1:
                        fig = px.bar(
                            x=list(range(len(w))),
                            y=w,
                            labels={'x':'Index','y':'Bias value'},
                            title=title
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # 3) n-D tensors → histogram of values
                    else:
                        flat = w.flatten()
                        fig = px.histogram(
                            flat,
                            nbins=50,
                            labels={'value':'Parameter value','count':'Frequency'},
                            title=title
                        )
                        st.plotly_chart(fig, use_container_width=True)

def show_test_results(model,result_df, accuracy_threshold):
    with st.expander("LSTM Model Test Results"):
        st.write("LSTM model results and performance metrics:")
        # use MAE, MSE, RMSE, and R2 score, MAPE and accuracy
        mae = mean_absolute_error(result_df['Actual'], result_df['Predicted'])
        mse = mean_squared_error(result_df['Actual'], result_df['Predicted'])
        rmse = np.sqrt(mse)
        r2 = r2_score(result_df['Actual'], result_df['Predicted'])
        mape = mean_absolute_percentage_error(result_df['Actual'], result_df['Predicted'])
        # Assuming a threshold of 0.5% for accuracy
        threshold = accuracy_threshold / 100
        preds = result_df['Predicted'].values
        actuals = result_df['Actual'].values
        # Calculate accuracy based on the threshold
        accuracy = np.mean(np.abs(preds - actuals) <= threshold * actuals)
        # st.write(f"Accuracy: {accuracy:.2%}")
        # from MAPE also calculate accuracy like 1- mape
        # Display items in rows of 3 columns each
        items = [
            ("Mean Absolute Error (MAE)", f"{mae:.2f}"),
            ("Root Mean Squared Error (RMSE)", f"{rmse:.2f}"),
            ("R2 Score", f"{r2:.2f}"),
            ("Mean Absolute Percentage Error (MAPE)", f"{mape:.2%}"),
            ("Accuracy from MAPE", f"{1 - mape:.2%}"),
            ("Accuracy (within threshold)", f"{accuracy:.2%}")
        ]
        for i in range(0, len(items), 3):
            cols = st.columns(3)
            for j, (label, value) in enumerate(items[i:i+3]):
                cols[j].metric(label, value)



def forecast_future(model, scaled_data, time_step, forecast_days, scaler, last_date):
    """
    Forecast future stock prices using the trained LSTM model.
    """
    last_seq = scaled_data[-time_step:].reshape((1, time_step, 1))
    forecast = []
    for _ in range(forecast_days):
        pred = model.predict(last_seq)
        forecast.append(pred[0, 0])
        last_seq = np.append(last_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
    forecast_vals = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast_df = pd.DataFrame(forecast_vals, index=future_index, columns=['Forecast'])
    return forecast_df


# Main Streamlit app
def main():
    st.title("Stock Price Analysis & LSTM Prediction")
    st.sidebar.markdown("#### Select Parameters ")

    # Load ticker list based on country selection
    country = st.sidebar.selectbox("Select Country", ['India', 'US', 'Canada'])
    tickers_df = load_ticker_list(country)


    plt.style.use("fivethirtyeight")
    plt.rcParams["figure.figsize"] = (10, 3)  # reduced height for figures
    ticker_names = tickers_df['Name'].sort_values().tolist()
    selected_name = st.sidebar.selectbox("Select Stock", ticker_names)
    symbol = find_symbol(tickers_df, selected_name)

    if not symbol:
        st.sidebar.error("Symbol not found for selected stock.")
        return

    # Display stock profile
    display_stock_profile(symbol, selected_name)

    period = st.sidebar.selectbox("Period", ["1y", "2y", "5y", "max"])
    # time_step = st.sidebar.slider("Past Days For Prediction", min_value=10, max_value=100, value=60)

    # Analysis options
    time_step, show_ma, moving_average, show_rsi, rsi_days, show_volatility, volatility_days, forecast_days, run_analysis, accuracy_threshold = get_analysis_options()

    if run_analysis:
        # Data loading
        # period = st.sidebar.selectbox("Period", ["1y", "2y", "5y", "max"])
        df = load_data(symbol, period)
        if df.empty:
            st.error('No data fetched. Check symbol or period.')
            return


        # Raw closing prices
        with st.expander(f"{selected_name} Prices", expanded=True):
            st.line_chart(df['Close'])

        # Performance metrics
        cum_ret, avg_ret, vol, sharpe = compute_metrics(df)
        st.subheader("Stock Performance")
        with st.expander(f"Stock Performance over {period} Details"):
            # write in 2*2 grid
            items = [
                ("Cumulative Return", f"{cum_ret:.2%}"),
                ("Average Daily Return", f"{avg_ret:.2%}"),
                ("Annualized Volatility", f"{vol:.2%}"),
                ("Sharpe Ratio", f"{sharpe:.2f}")
            ]
            for i in range(0, len(items), 2):
                cols = st.columns(2)
                for j, (label, value) in enumerate(items[i:i+2]):
                    cols[j].metric(label, value)

        # Moving averages
        if show_ma:
            with st.expander("Moving Average Details"):
                df[f'{moving_average}_MA'] = df['Close'].rolling(window=moving_average).mean()
                cols = ['Close', f'{moving_average}_MA']
                st.line_chart(df[cols].dropna())

        # RSI
        if show_rsi:
            with st.expander(f"RSI ({rsi_days}-day)"):
                df['RSI'] = calculate_rsi(df['Close'], rsi_days)
                st.line_chart(df['RSI'].dropna())

        # Volatility
        if show_volatility:
            with st.expander(f"Volatility ({volatility_days}-day)"):
                df['Volatility'] = df['Close'].pct_change().rolling(window=volatility_days).std() * np.sqrt(252)
                st.line_chart(df['Volatility'].dropna())
        

        # Prepare for LSTM training
        scaled_data, scaler = preprocess_data(df)
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = build_model((X.shape[1], 1))
        st.info("Training LSTM model... this may take a few minutes.")
        history = model.fit(X, y, batch_size=64, epochs=5, verbose=0)

        # Predictions
        predictions = model.predict(X)
        preds = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(y.reshape(-1, 1))

        result_df = pd.DataFrame({
            'Actual': actual.flatten(),
            'Predicted': preds.flatten()
        }, index=df.index[time_step:])

        with st.expander("Actual vs Predicted Prices", expanded=True):
            st.line_chart(result_df)

        # Show test results
        show_test_results(model,result_df, accuracy_threshold)

        # Training metrics show
        show_train_results(model, history)

        # Forecast future prices
        st.subheader("Forecasting Future Prices")
        forecast_df = forecast_future(model, scaled_data, time_step, forecast_days, scaler, df.index[-1])
        with st.expander(f"LSTM Model Forecasting for {forecast_days} Future Days", expanded=True):
            st.dataframe(forecast_df)
            
        # also add the forecasted values to the result_df for comparison
        result_df = pd.concat([result_df, forecast_df], axis=1)

        with st.expander("LSTM Model Forecasted Prices"):
            st.line_chart(result_df[['Actual', 'Forecast']])


if __name__ == "__main__":
    main()
