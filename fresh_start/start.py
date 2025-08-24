import streamlit as st
from coingecko import CoinGeckoHandler
import plotly.graph_objects as go


st.title("ALGO TRADING")

process = st.sidebar.selectbox(
    "Pair Trading bot (UI)",
    ( "Selecting Assests", "Backtesting", "Trading")
)

if process == "Selecting Assests":
    st.header("Selecting Assests")
    tabs = st.tabs(['Formal Selection', 'Price Charts'])
    with tabs[0]:
        with st.expander("STEP 1: SELECT CRYPTO"):
            sub_stage = st.selectbox(
            "Step 1: Choose Stage",
            (
                "Stage 1: Pre-filtering",
                "Stage 2: Grouping by Fundamental Logic",
                "Stage 3: Grouping by Statistical Behaviour",
                "Stage 4: Ranking Framework",
                "Final Selection"
            )
        )

            # Stage 1: Pre-filtering
            if sub_stage == "Stage 1: Pre-filtering":
                st.subheader("Pre-filtering Criteria")

                st.markdown("Apply pre-filtering based on the following:")
                liquidity = st.slider("Liquidity Threshold (e.g., 24h volume in $M)", 0, 100, 10)
                age = st.slider("Minimum Age (months)", 1, 60, 12)
                price_stability = st.slider("Price Stability Index (0-1)", 0.0, 1.0, 0.5)
                availability = st.multiselect("Select Exchanges", ["Binance", "Coinbase", "Kraken", "Bitfinex"])
                data_quality = st.selectbox("Data Quality Rating", ["High", "Medium", "Low"])

                if st.button("Run Pre-filtering"):
                    st.success("Pre-filtering executed with selected parameters.")
                    st.write(f"Selected Liquidity Threshold: ${liquidity}M")
                    st.write(f"Minimum Age: {age} months")
                    st.write(f"Price Stability Index: {price_stability}")
                    st.write(f"Exchanges Selected: {availability}")
                    st.write(f"Data Quality: {data_quality}")

            elif sub_stage == "Stage 2: Grouping by Fundamental Logic":
                st.subheader("Stage 2: Grouping by Fundamental Logic")
                st.info("This will be implemented later.")

            elif sub_stage == "Stage 3: Grouping by Statistical Behaviour":
                st.subheader("Stage 3: Grouping by Statistical Behaviour")
                st.info("This will be implemented later.")

            elif sub_stage == "Stage 4: Ranking Framework":
                st.subheader("Stage 4: Ranking Framework")

                st.markdown("Define weights for scoring metrics:")
                weight_volume = st.slider("Weight: Liquidity", 0, 100, 30)
                weight_age = st.slider("Weight: Longevity", 0, 100, 20)
                weight_stability = st.slider("Weight: Price Stability", 0, 100, 25)
                weight_quality = st.slider("Weight: Data Quality", 0, 100, 25)

                if st.button("Compute Scores"):
                    st.success("Scoring and ranking computed.")
                    st.write("This section will calculate a final score based on the weighted metrics.")
                    # Placeholder for score computation logic

            elif sub_stage == "Final Selection":
                st.subheader("Final Selection of Candidates")
                st.write("Display selected and ranked candidates here based on filters and scores.")
                st.info("This will summarize the final selection for Cointegration testing.")

    with tabs[1]:
        cg = CoinGeckoHandler()
        available_assets = [
            "bitcoin", "ethereum", "ripple", "litecoin", "monero", "cardano", "polkadot", "solana"
        ]

        # Sidebar selection
        selected_assets = st.multiselect("Select Cryptos", available_assets, default=["bitcoin"])

        # Loop through selected assets
        for asset_id in selected_assets:
            st.divider()
            col1, col2 = st.columns([1, 2])

            # ----- Metadata -----
            with col1:
                st.subheader(asset_id.capitalize())

                info = cg.get_basic_info(asset_id)
                scores = cg.get_coin_scores(asset_id)
                image_url = cg.get_coin_image(asset_id).get('thumb', None)
                launch_date = info.get('genesis_date', 'N/A')
                homepage = info.get('links', {}).get('homepage', [''])[0]

                if image_url:
                    st.image(image_url, width=64)

                st.write(f"**Name:** {info['name']}")
                st.write(f"**Symbol:** {info['symbol'].upper()}")
                st.write(f"**Market Cap Rank:** {scores.get('market_cap_rank', 'N/A')}")
                st.write(f"**Genesis Date:** {launch_date}")
                st.write(f"[Homepage]({homepage})")

            # ----- Price Chart -----
            with col2:
                st.subheader("Price Chart (365 days)")
                df = cg.get_price_history(asset_id, days=365)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['price'], mode='lines', name=asset_id
                ))
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

