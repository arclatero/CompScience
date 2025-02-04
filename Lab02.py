
# Sample dataset with predefined values
data = [
    {'age': 'Young', 'income': 'High', 'buys_computer': 'Yes'},
    {'age': 'Senior', 'income': 'Medium', 'buys_computer': 'Yes'},
    {'age': 'Senior', 'income': 'Low', 'buys_computer': 'No'},
    {'age': 'Middle-aged', 'income': 'Low', 'buys_computer': 'Yes'},
    {'age': 'Young', 'income': 'Medium', 'buys_computer': 'No'},
    {'age': 'Young', 'income': 'Low', 'buys_computer': 'No'},
    {'age': 'Senior', 'income': 'High', 'buys_computer': 'Yes'},
    {'age': 'Middle-aged', 'income': 'Medium', 'buys_computer': 'Yes'},
    {'age': 'Middle-aged', 'income': 'High', 'buys_computer': 'Yes'},
]

# Function to calculate prior probabilities
def prior_probabilities(data):
    class_counts = {}
    total_samples = len(data)
    for item in data:
        label = item['buys_computer']
        class_counts[label] = class_counts.get(label, 0) + 1
    return {cls: count / total_samples for cls, count in class_counts.items()}

# Function to calculate conditional probabilities
def likelihood(data, feature, value, given_class):
    count_feature_class = sum(1 for item in data if item[feature] == value and item['buys_computer'] == given_class)
    count_class = sum(1 for item in data if item['buys_computer'] == given_class)
    unique_values = len(set(item[feature] for item in data))
    return (count_feature_class + 1) / (count_class + unique_values)  # Laplace smoothing

# Naive Bayes classification function
def naive_bayes(test_sample, data):
    prior_probs = prior_probabilities(data)
    classes = prior_probs.keys()
    probabilities = {}
    for cls in classes:
        probabilities[cls] = prior_probs[cls]
        for feature, value in test_sample.items():
            if feature != 'buys_computer':
                probabilities[cls] *= likelihood(data, feature, value, cls)
    return max(probabilities, key=probabilities.get)

# Accept user input
valid_ages = {'Young', 'Middle-aged', 'Senior'}
valid_incomes = {'High', 'Medium', 'Low'}

def get_input():
    while True:
        try:
            age = input("Enter age (Young, Middle-aged, Senior): ").strip()
            if age not in valid_ages:
                raise ValueError("Invalid input. Choose from: Young, Middle-aged, Senior.")
            income = input("Enter income (High, Medium, Low): ").strip()
            if income not in valid_incomes:
                raise ValueError("Invalid input. Choose from: High, Medium, Low.")
            return {'age': age, 'income': income}
        except ValueError as e:
            print(e)

# Run prediction
user_data = get_input()
prediction = naive_bayes(user_data, data)
print(f"Prediction for input {user_data}: {prediction}")
