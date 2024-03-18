from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model4.pkl', 'rb'))
#scaler = pickle.load(open('scaler11.pkl', 'rb'))


app = Flask(__name__)



@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    # # Get the input data as dictionary
    # input_data = request.get_json()
    #
    # # Convert the input data to a numpy array
    # input_data_array = np.array([input_data['Pregnancies'], input_data['Glucose'], input_data['BloodPressure'],
    #                              input_data['SkinThickness'], input_data['Insulin'], input_data['BMI'],
    #                              input_data['DiabetesPedigreeFunction'], input_data['Age']])
    #
    # # Reshape the array for prediction
    # input_data_reshaped = input_data_array.reshape(1, -1)
    #
    # # Scale the input data
    # input_data_scaled = scaler.transform(input_data_reshaped)
    #
    # # Make the prediction
    # prediction = model.predict(input_data_scaled)
    #
    # # Get the prediction result
    # if prediction[0] == 1:
    #     result = 'Diabetic'
    # else:
    #     result = 'Not Diabetic'
    #
    # # Return the result as a JSON response
    # print(result)
    #return jsonify({'result': result})

    # Get the input data from the request
    # Pregnancies = request.form.get('Pregnancies')
    # Glucose = request.form.get('Glucose')
    # BloodPressure = request.form.get('BloodPressure')
    # SkinThickness = request.form.get('SkinThickness')
    # Insulin = request.form.get('Insulin')
    # BMI = request.form.get('BMI')
    # DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
    # Age = request.form.get('Age')
    #
    # input_data = np.array(
    #     [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    #
    # # make a prediction using the model
    # prediction = model.predict(input_data)

    # Convert the input data to a numpy array and reshape it
    # input_data = np.array(
    #     [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    # input_data_reshaped = input_data.reshape(1, -1)
    #
    # # Standardize the input data
    # std_data = scaler.transform(input_data_reshaped)
    #
    # # Make a prediction using the model
    # prediction = model.predict(std_data)[0]

    # Return the prediction as a JSON response
    # if prediction[0] == 0:
    #     output = {'diabetes': '0', 'message': 'The person is not diabetic'}
    # else:
    #     output = {'diabetes': '1', 'message': 'The person is diabetic'}
    #
    # return jsonify(output)


    Pregnancies = request.form.get('Pregnancies')
    Glucose = request.form.get('Glucose')
    BloodPressure = request.form.get('BloodPressure')
    SkinThickness = request.form.get('SkinThickness')
    Insulin = request.form.get('Insulin')
    BMI = request.form.get('BMI')
    DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
    Age = request.form.get('Age')

    input_data_as_numpy_array = np.asarray(
        (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age))
    # input_query = np.array(
    #     [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    input_query = input_data_as_numpy_array.reshape(1, -1)
    result = model.predict(input_query)[0]
    print(result)
    # result = {'Pregnancies': Pregnancies, 'Glucose': Glucose, 'BloodPressure': BloodPressure,
    #           'SkinThickness': SkinThickness, 'Insulin': Insulin,
    #           'BMI': BMI, 'DiabetesPedigreeFunction': DiabetesPedigreeFunction, 'Age': Age}

    return jsonify({'diabetes': str(result)})


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')
