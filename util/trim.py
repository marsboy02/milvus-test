import ast
import re


def remove_escape_sequences(data):
    cleaned_data = re.sub(r'\\\'', "'", data)
    cleaned_data = re.sub(r'\\\\', '\\', cleaned_data)
    cleaned_data_list = ast.literal_eval(cleaned_data)

    return cleaned_data_list
