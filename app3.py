import pickle
import sklearn

vectorizer_path = r"E:\NLP(Sentimate analysis project)\tfidf_vectorizer.pkl"
model_path = r"E:\NLP(Sentimate analysis project)\random_forest_model.pkl"

# Load using an old format
with open(vectorizer_path, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Re-save in the new format
with open("new_tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

with open("new_random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Pickle files saved in a new format!")
