# Importing functions from preprocess.py and knn_model.py
from preprocess import load_data, preprocess_text
from knn_model import train_knn, predict_emotion

def main():
    # Step 1: Load dataset
    print("Loading dataset...")
    texts, emotions = load_data(r"dataset\emotions.csv")  # Path to the dataset file
    
    # Step 2: Preprocess text data
    print("Preprocessing text data...")
    X, vectorizer = preprocess_text(texts)  # Vectorize the text data
    
    # Step 3: Train the KNN model
    print("Training KNN model...")
    knn = train_knn(X, emotions, n_neighbors=3)  # Train the KNN classifier
    
    print("\nSystem is ready! Enter a sentence to detect emotions.\n")
    
    # Step 4: Real-Time Emotion Detection
    while True:
        user_input = input("Enter a sentence (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting... Goodbye!")
            break
        
        # Predict emotion using the trained KNN model
        detected_emotion = predict_emotion(knn, vectorizer, user_input)
        print(f"Detected Emotion: {detected_emotion}")
        
        # Task recommendation based on emotion
        recommendations = {
            "Happy": "Focus on creative tasks or collaborate with your team.",
            "Stressed": "Take a break or prioritize your most critical tasks.",
            "Sad": "Consider talking to someone or doing a relaxing activity.",
            "Motivated": "Work on high-priority or challenging tasks.",
            "Burnout": "Inform HR and consider taking a wellness day."
        }
        print(f"Recommended Task: {recommendations.get(detected_emotion, 'No recommendation available.')}\n")

if __name__ == "_main_":
    main()