
# =========================================================
# AuroraStay Hotels - Streamlit Frontend
# =========================================================

import os
import streamlit as st
import requests


# =========================================================
# Configuration
# =========================================================

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:5000"
).rstrip("/")


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="AuroraStay Hotels",
    page_icon="🏨",
    layout="wide"
)


# =========================================================
# Sidebar - Retrieval parameters
# =========================================================

with st.sidebar:

    st.header("⚙️ RAG Parameters")

    k = st.slider(
        "Number of chunks (k)",
        min_value=1,
        max_value=20,
        value=3,
        help="Number of policy chunks retrieved from FAISS."
    )

    st.info(
        "The baseline evaluation used k=3. "
        "Higher values may improve coverage but can "
        "also introduce irrelevant information."
    )

    st.caption(
        "The language model settings are configured "
        "in the Flask backend."
    )


# =========================================================
# Main application
# =========================================================

st.title("🏨 AuroraStay Hotels")

st.write(
    "Ask questions about AuroraStay hotel policies, "
    "including reservations, cancellations, refunds, "
    "parking, pets, check-in times and guest services."
)


# =========================================================
# Response mode
# =========================================================

response_type = st.radio(
    "Choose response mode",
    [
        "Answer only",
        "Answer + Relevant Chunks"
    ],
    horizontal=True
)


# =========================================================
# Question input
# =========================================================

query = st.text_input(
    "Enter your hotel policy question",
    placeholder=(
        "e.g., What are the parking fees and "
        "check-in times at ASH-ATL Midtown?"
    )
)


# =========================================================
# Submit question
# =========================================================

if st.button("Submit", type="primary"):

    if not query.strip():

        st.warning("Please enter a hotel policy question.")

    else:

        # Select the correct Flask endpoint
        if response_type == "Answer only":
            endpoint = "/v1/answer"
        else:
            endpoint = "/v1/answer_with_relevant_chunks"

        # Build request payload
        payload = {
            "query": query.strip(),
            "k": k
        }

        try:

            with st.spinner("Searching AuroraStay policies..."):

                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json=payload,
                    timeout=120
                )

            # Successful API response
            if response.status_code == 200:

                result = response.json()

                st.subheader("Answer")

                st.write(
                    result.get(
                        "answer",
                        "No answer was returned."
                    )
                )

                # Display retrieved chunks when requested
                if response_type == "Answer + Relevant Chunks":

                    st.subheader("Relevant Policy Chunks")

                    chunks = result.get(
                        "relevant_chunks",
                        []
                    )

                    if chunks:

                        for i, chunk in enumerate(chunks, start=1):

                            with st.expander(f"Chunk {i}"):
                                st.text(chunk)

                    else:

                        st.info(
                            "No relevant policy chunks were returned."
                        )

                st.caption(
                    "Policy information may require verification "
                    "against the current property profile or "
                    "reservation details before being quoted "
                    "to a guest."
                )

            # API returned an error
            else:

                st.error(
                    f"Backend API returned HTTP "
                    f"{response.status_code}."
                )

                try:
                    st.write(response.json())
                except ValueError:
                    st.write(response.text[:500])

        except requests.exceptions.Timeout:

            st.error(
                "The request timed out. "
                "Please try again."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to the AuroraStay "
                "Flask backend. Please check that "
                "the backend service is running."
            )

        except requests.exceptions.RequestException as error:

            st.error(
                f"An API request error occurred: {error}"
            )
