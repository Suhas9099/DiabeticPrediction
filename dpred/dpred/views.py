from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    result1 = None
    input_values = {}

    if request.method == "GET":
        df = pd.read_csv(r"C:\Users\Kpard\Downloads\diabetes.csv")
        X = df.drop(columns='Outcome', axis=1)
        Y = df['Outcome']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        classifier = LogisticRegression()
        classifier.fit(X_train, Y_train)

        input_values = {
            'n1': request.GET.get('n1', ''),
            'n2': request.GET.get('n2', ''),
            'n3': request.GET.get('n3', ''),
            'n4': request.GET.get('n4', ''),
            'n5': request.GET.get('n5', ''),
            'n6': request.GET.get('n6', ''),
            'n7': request.GET.get('n7', ''),
            'n8': request.GET.get('n8', ''),
        }

        try:
            input_data = [
                float(input_values['n1']),
                float(input_values['n2']),
                float(input_values['n3']),
                float(input_values['n4']),
                float(input_values['n5']),
                float(input_values['n6']),
                float(input_values['n7']),
                float(input_values['n8'])
            ]

            # Scale the input data
            input_data = scaler.transform([input_data])

            pred = classifier.predict(input_data)

            result1 = "Positive" if pred == [1] else "Negative"
        except ValueError:
            result1 = ""

    return render(request, "predict.html", {"result2": result1, "input_values": input_values})
