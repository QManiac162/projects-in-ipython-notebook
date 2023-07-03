import streamlit as sl
import pickle

# loading the model to predict data
pickle_in = open('RFC.pkl', 'rb')
RFC = pickle.load(pickle_in)

def welcome():
    return '! YOUR PRESENCE IS DETECTED !'

# creating the function which will do the prediction from user inputs
def classification(sepal_length, sepal_width, petal_length, petal_width):
    classification = RFC.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    print(classification)
    return classification

# the main function which defines the webpage
def main():
    html_temp = """
    <div style = "background-color: cyan; padding: 13px">
    <h1 style = "color: black; text-align: centre;"> IRIS-SPECIES-CLASSIFICATION </h1>
    """

    sl.markdown(html_temp, unsafe_allow_html = True)
    sepal_length = sl.text_input("SEPAL LENGTH")
    sepal_width = sl.text_input("SEPAL WIDTH")
    petal_length = sl.text_input("PETAL LENGTH")
    petal_width = sl.text_input("PETAL WIDTH")
    result = ""

    if sl.button("PREDICT"):
        result = classification(sepal_length, sepal_width, petal_length, petal_width)
    sl.success("{}" .format(result))

if __name__=='__main__':
    main()
