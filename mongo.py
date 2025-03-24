# mongodb_utils.py
from pymongo import MongoClient
import streamlit as st

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"  # 
DATABASE_NAME = "mind_body_wellness"
COLLECTION_NAME = "user_data"

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

def save_to_mongodb(data):
    """Saves user data to MongoDB."""
    try:
        collection.insert_one(data)
        st.success("✅ User data saved successfully to MongoDB!")
    except Exception as e:
        st.error(f"An error occurred while saving to MongoDB: {e}")
