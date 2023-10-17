import os
def create_folders_if_not_exist():
    # Create a 'models' folder if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Create a 'data' folder if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')