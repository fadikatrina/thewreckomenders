from helpers_data_load import nutrition_data
from ast import literal_eval


def apply_health_filter(recipe_ids, limits=[.15, .35, .10, .25]):
    """
    :param limits: max sugar amount, max sodium, min protein amount, max saturated_fat
    (not in grams, but percentage of nutritional content)
    :return: recipe ids
    """
    healthy_recipes = []

    # Get the nutritional information for the relevant recipes
    recipes = nutrition_data.loc[nutrition_data['id'].isin(recipe_ids)]

    for index, recipe in recipes.iterrows():
        # Nutrition information in calories, total fat, sugar, sodium, protein, saturated fat, carbohydrates
        nutrition_values = recipe['nutrition']
        # convert string version of array into a proper array
        nutrition_values = literal_eval(nutrition_values)

        sugar = nutrition_values[2]
        sodium = nutrition_values[3]
        protein = nutrition_values[4]
        saturated_fat = nutrition_values[5]

        # Since the nutritional info is in absolute numbers instead of per 100 grams, we'll normalize
        normalization_factor = sum(nutrition_values[1:])
        normalization_factor = max(normalization_factor, 0.01)

        sugar /= normalization_factor
        sodium /= normalization_factor
        protein /= normalization_factor
        saturated_fat /= normalization_factor

        if sugar < limits[0] and sodium < limits[1] and protein > limits[2] and saturated_fat < limits[3]:
            healthy_recipes.append(recipe['id'])

    return healthy_recipes
