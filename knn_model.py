from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def predict_emotion(knn, vectorizer, new_text):
    X_new = vectorizer.transform([new_text])
    return knn.predict(X_new)[0]