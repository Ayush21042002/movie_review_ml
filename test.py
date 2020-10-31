import pickle

# load the saved model
model = pickle.load(open("text_clf", 'rb'))

review = input("Enter the review of a movie")

pd = model.predict([review])

print(pd[0])