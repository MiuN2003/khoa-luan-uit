import streamlit as st
from define_model import *
from predictor import *
from preprocessing import *
from retrieval import ER_BM25

# Define the Streamlit interface
def main():
    st.title("Text Claim Prediction")

    # Input field for the claim
    claim = st.text_input("Enter the claim", "Việt Nam là quốc gia đông dân thứ 3 thế giới")

    # Get content base on the claim
    content = ER_BM25(claim, 7)

    # Display the content
    st.write("Content:")
    st.write(content)

    # Allow the user to edit content
    content = st.text_area("Edit the content", content)
    
    if st.button("Predict"):
        # Get the dataset using the entered content and claim
        test = get_dataset(content=content, claim=claim)

        # Display the input data
        st.write(f"Content: {content}")
        st.write(f"Claim: {claim}")

        # Run the prediction
        test_results = predict(test)

        # Display the results
        st.write("Prediction Results:")
        for result in test_results:
            st.write(result)

# Run the Streamlit app
if __name__ == "__main__":
    main()
