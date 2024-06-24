import torch
import os
import numpy as np
import paths as p

""" This module performs comparison on vector distances to closest k cluster. """

def get_image_vectors():
    """ Loads raw images and transforms them into vectors using trained model """
    # import from vector extractor
    pass


def load_images(folder_path):
    """ Function to create a dictionary of all vectors from a specific folder.

    Args:
    folder_path: str
    Path to the folder where vectors exist

    Returns:
    Dictionary with class names as key and image vectors as items """

    image_vector_dict = {}

    for folder_name in os.listdir(folder_path):
        class_level_result_dict = {}
        folder_path_full = os.path.join(folder_path, folder_name)

        for file_name in os.listdir(folder_path_full):
            file_path = os.path.join(folder_path_full, file_name)

            # Extract the name of the numpy array
            array_name = os.path.splitext(file_name)[0]

            # Load the numpy array
            numpik_loaded = np.load(file_path)

            # Add numpy array to class_level_result_dict with the file name as the key
            class_level_result_dict[array_name] = numpik_loaded

        # Add class_level_result_dict to complete_result_dict with the folder name as the key
        image_vector_dict[folder_name] = class_level_result_dict

    return image_vector_dict


def calculate_distances(image_dict_1, image_dict_2):
    """
    Function which loops through dictionaries and calculates distances between two vectors.

    Args:
    image_dict_1: dict
    Dictionary containing vectors.

    image_dict_2: dict
    Dictionary containing vectors.

    Returns: dict
    Dictionary with distances between 2 vectors.
    The structure of the dictionary is:
    dict = {Class 1: {image_name: {Class 2: {image_name : vector_distance}}}}
    """
    distances = {}

    for class_name_1, items_dict_1 in image_dict_1.items():
        distances[class_name_1] = {}

        for file_name_1, items_1 in items_dict_1.items():
            distances[class_name_1][file_name_1] = {}

            for class_name_2, items_dict_2 in image_dict_2.items():
                distances[class_name_1][file_name_1][class_name_2] = {}
                # if class_name_1 == class_name_2:
                #     # Skip distance calculation if class_name_1 equals class_name_2
                #     continue

                for file_name_2, items_2 in items_dict_2.items():
                    if file_name_1 == file_name_2:
                        continue

                    distance = np.linalg.norm(np.array(items_1) - np.array(items_2))
                    distances[class_name_1][file_name_1][class_name_2][file_name_2] = distance

    return distances


def recursive_sort(dictionary):
    sorted_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            # Recursively sort nested dictionaries
            sorted_dict[key] = recursive_sort(value)
        else:
            sorted_dict[key] = value

    # Sort the dictionary at this level by values if they are all numeric
    if all(isinstance(v, (int, float)) for v in sorted_dict.values()):
        sorted_dict = dict(sorted(sorted_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict

def find_top_n_min_values(data, num_top_values=5):
    """
    Function performing K nearest neighbours on a dictionary.

    Args:

    data: dict
    Dictionary with all classes and their distances.
    The expected structure is: {C1: {IMG1: {C2: IMG1: X, IMG2: X...}, IMG2: {C3: X...}}}
    The function 'calculate_distances' prepares the data in this expected format.

    num_top_values: int
    Number of top k values to be returned.

    Returns: dict

    """

    top_min_values = {}

    def extract_numeric_values(d, class_name, path=[]):
        if isinstance(d, dict):
            for key, value in d.items():
                new_path = path + [key]
                if isinstance(value, dict):
                    extract_numeric_values(value, class_name, new_path)
                else:
                    try:
                        top_min_values.setdefault(class_name, []).append((new_path[-1], float(value)))
                    except (TypeError, ValueError):
                        pass

    for class_name, class_data in data.items():
        extract_numeric_values(class_data, class_name)

    # Sort and find the top N minimal values for each class
    for class_name, values in top_min_values.items():
        sorted_values = sorted(values, key=lambda x: x[1])[:num_top_values]
        class_output = {}
        for i, (key, value) in enumerate(sorted_values, start=1):
            class_output[i] = {class_name: {key: value}}
        top_min_values[class_name] = class_output

    return top_min_values


sample = {'AcornSquash': {'n07717410_141': {'AcornSquash': {'n07717410_21': 0.1, 'n07717410_35': 0.2, 'n07717410_43': 0.3,'n07717410_214': 0.4, 'n07717410_355': 0.5, 'n07717410_436': 0.6,'n07717410_217': 0.7, 'n07717410_358': 0.8, 'n07717410_439': 0.9},
                                            'Avocado': {'n07764847_114': 0.3, 'n07764847_17': 0.3, 'n07764847_210': 0.3},
                                            'Berry': {'n07743224_103': 0.1, 'n07743224_113': 0.1, 'n07743224_146': 0.1}},
                          'n0892312_21': {'AcornSquash': {'n07717410_21': 0.1, 'n07717410_35': 0.1, 'n07717410_43': 0.1},
                                            'Avocado': {'n07764847_114': 0.2, 'n07764847_17': 0.2, 'n07764847_210': 0.2},
                                            'Berry': {'n07743224_103': 0.3, 'n07743224_113': 0.3, 'n07743224_146': 0.3}}},
 'Avocado': {'n07717410_141': {'AcornSquash': {'n07717410_21': 0.4, 'n07717410_35': 0.4, 'n07717410_43': 0.4},
                                            'Avocado': {'n07764847_114': 0.4, 'n07764847_17': 0.4, 'n07764847_210': 0.4},
                                            'Berry': {'n07743224_103': 0.5, 'n07743224_113': 0.5, 'n07743224_146': 0.5}},
                          'n0892312_21': {'AcornSquash': {'n07717410_21': 0.1, 'n07717410_35': 0.1, 'n07717410_43': 0.1},
                                            'Avocado': {'n07764847_114': 0.2, 'n07764847_17': 0.2, 'n07764847_210': 0.2},
                                            'Berry': {'n07743224_103': 0.3, 'n07743224_113': 0.3, 'n07743224_146': 0.3}}}}

if __name__ == '__main__':
    baseline_image_vectors_1 = load_images("D:/PhD/Images/Vectors/Sample1")
    baseline_image_vectors_2 = load_images("D:/PhD/Images/Vectors/Sample2")
    print("Images Loaded")
    dict_to_compare = calculate_distances(baseline_image_vectors_1, baseline_image_vectors_2)
    print(dict_to_compare)
    rec_sort = recursive_sort(dict_to_compare)
    rec_sort = recursive_sort(sample2)
    print("Sorted dictionary")
    print(rec_sort)
    top_min_values = find_top_n_min_values(sample)
    #print("Find top min values")
    #print(top_min_values)
    print("Adaptation analysis finished running.")


