import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def fuzzy_recommend_nutrients(age_input, weight_input, height_input):

    # Define the fuzzy variables and their ranges
    age = ctrl.Antecedent(np.arange(0, 101, 1), "age")
    weight = ctrl.Antecedent(np.arange(20, 201, 1), "weight")
    height = ctrl.Antecedent(np.arange(100, 251, 1), "height")

    # Define fuzzy output variables for nutrients
    protein = ctrl.Consequent(np.arange(0, 201, 1), "protein")
    fat = ctrl.Consequent(np.arange(0, 101, 1), "fat")
    carbohydrates = ctrl.Consequent(np.arange(0, 401, 1), "carbohydrates")
    magnesium = ctrl.Consequent(np.arange(0, 1001, 1), "magnesium")
    sodium = ctrl.Consequent(np.arange(0, 3001, 1), "sodium")
    potassium = ctrl.Consequent(np.arange(1000, 5001, 1), "potassium")
    saturated_fat = ctrl.Consequent(np.arange(0, 20001, 1), "saturated_fats")
    monounsaturated_fat = ctrl.Consequent(
        np.arange(0, 60001, 1), "monounsaturated_fats"
    )
    polyunsaturated_fat = ctrl.Consequent(
        np.arange(0, 20001, 1), "polyunsaturated_fats"
    )
    free_sugar = ctrl.Consequent(np.arange(0, 101, 1), "free_sugar")
    starch = ctrl.Consequent(np.arange(0, 151, 1), "starch")

    # Define membership functions for age, weight, height
    age["young"] = fuzz.trapmf(age.universe, [0, 0, 20, 30])
    age["middle_aged"] = fuzz.trapmf(age.universe, [25, 35, 50, 60])
    age["senior"] = fuzz.trapmf(age.universe, [55, 65, 100, 100])

    weight["underweight"] = fuzz.trapmf(weight.universe, [30, 30, 50, 60])
    weight["normal"] = fuzz.trapmf(weight.universe, [55, 65, 75, 85])
    weight["overweight"] = fuzz.trapmf(weight.universe, [80, 90, 110, 120])
    weight["obese"] = fuzz.trapmf(weight.universe, [115, 130, 200, 200])

    height["short"] = fuzz.trapmf(height.universe, [100, 100, 140, 155])
    height["average"] = fuzz.trapmf(height.universe, [150, 165, 175, 185])
    height["tall"] = fuzz.trapmf(height.universe, [180, 190, 250, 250])

    protein["low"] = fuzz.trapmf(protein.universe, [0, 0, 24, 140])
    protein["medium"] = fuzz.trapmf(protein.universe, [100, 150, 200, 250])
    protein["high"] = fuzz.trapmf(protein.universe, [240, 270, 300, 450])

    # Repeat for other nutrients for fat ----  --------
    fat["low"] = fuzz.trapmf(fat.universe, [0, 0, 20, 50])
    fat["medium"] = fuzz.trapmf(fat.universe, [40, 60, 70, 90])
    fat["high"] = fuzz.trapmf(fat.universe, [80, 100, 150, 200])

    # Repeat for other nutrients (fat, carbohydrates, etc.)
    carbohydrates["low"] = fuzz.trapmf(carbohydrates.universe, [0, 0, 50, 150])
    carbohydrates["medium"] = fuzz.trapmf(carbohydrates.universe, [100, 150, 200, 250])
    carbohydrates["high"] = fuzz.trapmf(carbohydrates.universe, [230, 300, 400, 500])

    # Example for fat:
    free_sugar["low"] = fuzz.trapmf(free_sugar.universe, [0, 0, 10, 25])
    free_sugar["medium"] = fuzz.trapmf(free_sugar.universe, [20, 30, 40, 50])
    free_sugar["high"] = fuzz.trapmf(free_sugar.universe, [45, 60, 75, 100])

    starch["low"] = fuzz.trapmf(starch.universe, [0, 0, 40, 80])
    starch["medium"] = fuzz.trapmf(starch.universe, [70, 90, 120, 150])
    starch["high"] = fuzz.trapmf(starch.universe, [140, 160, 200, 250])

    magnesium["low"] = fuzz.trapmf(magnesium.universe, [0, 0, 50, 100])
    magnesium["medium"] = fuzz.trapmf(magnesium.universe, [80, 150, 200, 300])
    magnesium["high"] = fuzz.trapmf(magnesium.universe, [250, 350, 400, 500])

    sodium["low"] = fuzz.trapmf(sodium.universe, [0, 0, 500, 1000])
    sodium["medium"] = fuzz.trapmf(sodium.universe, [800, 1200, 1800, 2300])
    sodium["high"] = fuzz.trapmf(sodium.universe, [2000, 2500, 3000, 4000])

    potassium["low"] = fuzz.trapmf(potassium.universe, [0, 0, 1500, 2000])
    potassium["medium"] = fuzz.trapmf(potassium.universe, [1800, 2500, 3000, 3500])
    potassium["high"] = fuzz.trapmf(potassium.universe, [3300, 4000, 5000, 6000])

    saturated_fat["low"] = fuzz.trapmf(saturated_fat.universe, [0, 0, 5000, 10000])
    saturated_fat["medium"] = fuzz.trapmf(
        saturated_fat.universe, [9000, 12000, 15000, 20000]
    )
    saturated_fat["high"] = fuzz.trapmf(
        saturated_fat.universe, [18000, 22000, 30000, 35000]
    )

    monounsaturated_fat["low"] = fuzz.trapmf(
        monounsaturated_fat.universe, [0, 0, 5000, 10000]
    )
    monounsaturated_fat["medium"] = fuzz.trapmf(
        monounsaturated_fat.universe, [9000, 12000, 15000, 20000]
    )
    monounsaturated_fat["high"] = fuzz.trapmf(
        monounsaturated_fat.universe, [18000, 22000, 30000, 35000]
    )

    polyunsaturated_fat["low"] = fuzz.trapmf(
        polyunsaturated_fat.universe, [0, 0, 5000, 10000]
    )
    polyunsaturated_fat["medium"] = fuzz.trapmf(
        polyunsaturated_fat.universe, [9000, 12000, 15000, 20000]
    )
    polyunsaturated_fat["high"] = fuzz.trapmf(
        polyunsaturated_fat.universe, [18000, 22000, 30000, 35000]
    )

    # Define the fuzzy rules
    rule1 = ctrl.Rule(
        age["young"] & weight["underweight"] & height["short"],
        (
            protein["low"],
            fat["low"],
            carbohydrates["medium"],
            magnesium["low"],
            sodium["low"],
            potassium["low"],
            saturated_fat["low"],
            monounsaturated_fat["low"],
            polyunsaturated_fat["low"],
            free_sugar["low"],
            starch["low"],
        ),
    )
    rule2 = ctrl.Rule(
        age["young"] & weight["underweight"] & height["average"],
        (
            protein["low"],
            fat["low"],
            carbohydrates["medium"],
            magnesium["low"],
            sodium["low"],
            potassium["medium"],
            saturated_fat["low"],
            monounsaturated_fat["low"],
            polyunsaturated_fat["low"],
            free_sugar["low"],
            starch["low"],
        ),
    )
    rule3 = ctrl.Rule(
        age["young"] & weight["underweight"] & height["tall"],
        (
            protein["low"],
            fat["low"],
            carbohydrates["medium"],
            magnesium["low"],
            sodium["low"],
            potassium["medium"],
            saturated_fat["low"],
            monounsaturated_fat["low"],
            polyunsaturated_fat["low"],
            free_sugar["low"],
            starch["medium"],
        ),
    )
    rule4 = ctrl.Rule(
        age["young"] & weight["normal"] & height["short"],
        (
            protein["medium"],
            fat["low"],
            carbohydrates["medium"],
            magnesium["medium"],
            sodium["medium"],
            potassium["medium"],
            saturated_fat["low"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["medium"],
        ),
    )
    rule5 = ctrl.Rule(
        age["young"] & weight["normal"] & height["average"],
        (
            protein["medium"],
            fat["medium"],
            carbohydrates["medium"],
            magnesium["medium"],
            sodium["medium"],
            potassium["medium"],
            saturated_fat["medium"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["medium"],
        ),
    )
    rule6 = ctrl.Rule(
        age["young"] & weight["normal"] & height["tall"],
        (
            protein["medium"],
            fat["medium"],
            carbohydrates["high"],
            magnesium["medium"],
            sodium["medium"],
            potassium["high"],
            saturated_fat["medium"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["high"],
        ),
    )
    rule7 = ctrl.Rule(
        age["young"] & weight["overweight"] & height["short"],
        (
            protein["high"],
            fat["medium"],
            carbohydrates["medium"],
            magnesium["medium"],
            sodium["high"],
            potassium["medium"],
            saturated_fat["high"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["medium"],
        ),
    )
    rule8 = ctrl.Rule(
        age["young"] & weight["overweight"] & height["average"],
        (
            protein["high"],
            fat["medium"],
            carbohydrates["medium"],
            magnesium["high"],
            sodium["high"],
            potassium["medium"],
            saturated_fat["high"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["medium"],
        ),
    )
    rule9 = ctrl.Rule(
        age["young"] & weight["overweight"] & height["tall"],
        (
            protein["high"],
            fat["medium"],
            carbohydrates["high"],
            magnesium["high"],
            sodium["high"],
            potassium["high"],
            saturated_fat["high"],
            monounsaturated_fat["high"],
            polyunsaturated_fat["high"],
            free_sugar["medium"],
            starch["high"],
        ),
    )
    rule10 = ctrl.Rule(
        age["young"] & weight["obese"] & height["short"],
        (
            protein["high"],
            fat["high"],
            carbohydrates["medium"],
            magnesium["high"],
            sodium["high"],
            potassium["medium"],
            saturated_fat["high"],
            monounsaturated_fat["high"],
            polyunsaturated_fat["high"],
            free_sugar["high"],
            starch["medium"],
        ),
    )
    rule11 = ctrl.Rule(
        age["young"] & weight["obese"] & height["average"],
        (
            protein["high"],
            fat["high"],
            carbohydrates["medium"],
            magnesium["high"],
            sodium["high"],
            potassium["medium"],
            saturated_fat["high"],
            monounsaturated_fat["high"],
            polyunsaturated_fat["high"],
            free_sugar["high"],
            starch["medium"],
        ),
    )
    rule12 = ctrl.Rule(
        age["young"] & weight["obese"] & height["tall"],
        (
            protein["high"],
            fat["high"],
            carbohydrates["high"],
            magnesium["high"],
            sodium["high"],
            potassium["high"],
            saturated_fat["high"],
            monounsaturated_fat["high"],
            polyunsaturated_fat["high"],
            free_sugar["high"],
            starch["high"],
        ),
    )
    rule13 = ctrl.Rule(
        age["middle_aged"] & weight["underweight"] & height["short"],
        (
            protein["low"],
            fat["low"],
            carbohydrates["low"],
            magnesium["low"],
            sodium["low"],
            potassium["low"],
            saturated_fat["low"],
            monounsaturated_fat["low"],
            polyunsaturated_fat["low"],
            free_sugar["low"],
            starch["low"],
        ),
    )
    rule14 = ctrl.Rule(
        age["middle_aged"] & weight["underweight"] & height["average"],
        (
            protein["low"],
            fat["low"],
            carbohydrates["low"],
            magnesium["low"],
            sodium["low"],
            potassium["low"],
            saturated_fat["low"],
            monounsaturated_fat["low"],
            polyunsaturated_fat["low"],
            free_sugar["low"],
            starch["low"],
        ),
    )
    rule15 = ctrl.Rule(
        age["middle_aged"] & weight["underweight"] & height["tall"],
        (
            protein["low"],
            fat["low"],
            carbohydrates["medium"],
            magnesium["low"],
            sodium["low"],
            potassium["medium"],
            saturated_fat["low"],
            monounsaturated_fat["low"],
            polyunsaturated_fat["low"],
            free_sugar["low"],
            starch["medium"],
        ),
    )
    rule16 = ctrl.Rule(
        age["middle_aged"] & weight["normal"] & height["short"],
        (
            protein["medium"],
            fat["low"],
            carbohydrates["medium"],
            magnesium["medium"],
            sodium["medium"],
            potassium["medium"],
            saturated_fat["low"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["medium"],
        ),
    )
    rule17 = ctrl.Rule(
        age["middle_aged"] & weight["normal"] & height["average"],
        (
            protein["medium"],
            fat["medium"],
            carbohydrates["medium"],
            magnesium["medium"],
            sodium["medium"],
            potassium["medium"],
            saturated_fat["medium"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["medium"],
        ),
    )
    rule18 = ctrl.Rule(
        age["middle_aged"] & weight["normal"] & height["tall"],
        (
            protein["medium"],
            fat["medium"],
            carbohydrates["high"],
            magnesium["medium"],
            sodium["medium"],
            potassium["high"],
            saturated_fat["medium"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["high"],
        ),
    )
    rule19 = ctrl.Rule(
        age["middle_aged"] & weight["overweight"] & height["short"],
        (
            protein["high"],
            fat["medium"],
            carbohydrates["medium"],
            magnesium["medium"],
            sodium["high"],
            potassium["medium"],
            saturated_fat["high"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["medium"],
        ),
    )
    rule20 = ctrl.Rule(
        age["middle_aged"] & weight["overweight"] & height["average"],
        (
            protein["high"],
            fat["medium"],
            carbohydrates["medium"],
            magnesium["high"],
            sodium["high"],
            potassium["medium"],
            saturated_fat["high"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["medium"],
        ),
    )
    rule21 = ctrl.Rule(
        age["middle_aged"] & weight["overweight"] & height["tall"],
        (
            protein["high"],
            fat["medium"],
            carbohydrates["high"],
            magnesium["high"],
            sodium["high"],
            potassium["high"],
            saturated_fat["high"],
            monounsaturated_fat["high"],
            polyunsaturated_fat["high"],
            free_sugar["medium"],
            starch["high"],
        ),
    )
    rule22 = ctrl.Rule(
        age["middle_aged"] & weight["obese"] & height["short"],
        (
            protein["high"],
            fat["high"],
            carbohydrates["medium"],
            magnesium["high"],
            sodium["high"],
            potassium["medium"],
            saturated_fat["high"],
            monounsaturated_fat["high"],
            polyunsaturated_fat["high"],
            free_sugar["high"],
            starch["medium"],
        ),
    )
    rule23 = ctrl.Rule(
        age["middle_aged"] & weight["obese"] & height["average"],
        (
            protein["high"],
            fat["high"],
            carbohydrates["medium"],
            magnesium["high"],
            sodium["high"],
            potassium["medium"],
            saturated_fat["high"],
            monounsaturated_fat["high"],
            polyunsaturated_fat["high"],
            free_sugar["high"],
            starch["medium"],
        ),
    )
    rule24 = ctrl.Rule(
        age["middle_aged"] & weight["obese"] & height["tall"],
        (
            protein["high"],
            fat["high"],
            carbohydrates["high"],
            magnesium["high"],
            sodium["high"],
            potassium["high"],
            saturated_fat["high"],
            monounsaturated_fat["high"],
            polyunsaturated_fat["high"],
            free_sugar["high"],
            starch["high"],
        ),
    )
    rule25 = ctrl.Rule(
        age["senior"] & weight["underweight"] & height["short"],
        (
            protein["low"],
            fat["low"],
            carbohydrates["low"],
            magnesium["low"],
            sodium["low"],
            potassium["low"],
            saturated_fat["low"],
            monounsaturated_fat["low"],
            polyunsaturated_fat["low"],
            free_sugar["low"],
            starch["low"],
        ),
    )
    rule26 = ctrl.Rule(
        age["senior"] & weight["underweight"] & height["average"],
        (
            protein["low"],
            fat["low"],
            carbohydrates["low"],
            magnesium["low"],
            sodium["low"],
            potassium["low"],
            saturated_fat["low"],
            monounsaturated_fat["low"],
            polyunsaturated_fat["low"],
            free_sugar["low"],
            starch["low"],
        ),
    )
    rule27 = ctrl.Rule(
        age["senior"] & weight["underweight"] & height["tall"],
        (
            protein["low"],
            fat["low"],
            carbohydrates["medium"],
            magnesium["low"],
            sodium["low"],
            potassium["medium"],
            saturated_fat["low"],
            monounsaturated_fat["low"],
            polyunsaturated_fat["low"],
            free_sugar["low"],
            starch["medium"],
        ),
    )
    rule28 = ctrl.Rule(
        age["senior"] & weight["normal"] & height["short"],
        (
            protein["medium"],
            fat["low"],
            carbohydrates["medium"],
            magnesium["medium"],
            sodium["medium"],
            potassium["medium"],
            saturated_fat["low"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["medium"],
        ),
    )
    rule29 = ctrl.Rule(
        age["senior"] & weight["normal"] & height["average"],
        (
            protein["medium"],
            fat["medium"],
            carbohydrates["medium"],
            magnesium["medium"],
            sodium["medium"],
            potassium["medium"],
            saturated_fat["medium"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["medium"],
        ),
    )
    rule30 = ctrl.Rule(
        age["senior"] & weight["normal"] & height["tall"],
        (
            protein["medium"],
            fat["medium"],
            carbohydrates["high"],
            magnesium["medium"],
            sodium["medium"],
            potassium["high"],
            saturated_fat["medium"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["high"],
        ),
    )
    rule31 = ctrl.Rule(
        age["senior"] & weight["overweight"] & height["short"],
        (
            protein["high"],
            fat["medium"],
            carbohydrates["medium"],
            magnesium["medium"],
            sodium["high"],
            potassium["medium"],
            saturated_fat["high"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["medium"],
        ),
    )
    rule32 = ctrl.Rule(
        age["senior"] & weight["overweight"] & height["average"],
        (
            protein["high"],
            fat["medium"],
            carbohydrates["medium"],
            magnesium["high"],
            sodium["high"],
            potassium["medium"],
            saturated_fat["high"],
            monounsaturated_fat["medium"],
            polyunsaturated_fat["medium"],
            free_sugar["medium"],
            starch["medium"],
        ),
    )
    rule33 = ctrl.Rule(
        age["senior"] & weight["overweight"] & height["tall"],
        (
            protein["high"],
            fat["medium"],
            carbohydrates["high"],
            magnesium["high"],
            sodium["high"],
            potassium["high"],
            saturated_fat["high"],
            monounsaturated_fat["high"],
            polyunsaturated_fat["high"],
            free_sugar["medium"],
            starch["high"],
        ),
    )
    rule34 = ctrl.Rule(
        age["senior"] & weight["obese"] & height["short"],
        (
            protein["high"],
            fat["high"],
            carbohydrates["medium"],
            magnesium["high"],
            sodium["high"],
            potassium["medium"],
            saturated_fat["high"],
            monounsaturated_fat["high"],
            polyunsaturated_fat["high"],
            free_sugar["high"],
            starch["medium"],
        ),
    )
    rule35 = ctrl.Rule(
        age["senior"] & weight["obese"] & height["average"],
        (
            protein["high"],
            fat["high"],
            carbohydrates["medium"],
            magnesium["high"],
            sodium["high"],
            potassium["medium"],
            saturated_fat["high"],
            monounsaturated_fat["high"],
            polyunsaturated_fat["high"],
            free_sugar["high"],
            starch["medium"],
        ),
    )
    rule36 = ctrl.Rule(
        age["senior"] & weight["obese"] & height["tall"],
        (
            protein["high"],
            fat["high"],
            carbohydrates["high"],
            magnesium["high"],
            sodium["high"],
            potassium["high"],
            saturated_fat["high"],
            monounsaturated_fat["high"],
            polyunsaturated_fat["high"],
            free_sugar["high"],
            starch["high"],
        ),
    )

    # Define more rules here...

    # Create the control system and simulation
    food_ctrl = ctrl.ControlSystem(
        [
            rule1,
            rule2,
            rule3,
            rule4,
            rule5,
            rule6,
            rule7,
            rule8,
            rule9,
            rule10,
            rule11,
            rule12,
            rule13,
            rule14,
            rule15,
            rule16,
            rule17,
            rule18,
            rule19,
            rule20,
            rule21,
            rule22,
            rule23,
            rule24,
            rule25,
            rule26,
            rule27,
            rule28,
            rule29,
            rule30,
            rule31,
            rule32,
            rule33,
            rule34,
            rule35,
            rule36,
        ]
    )
    food_simulation = ctrl.ControlSystemSimulation(food_ctrl)

    # Input the values into the simulation
    food_simulation.input["age"] = age_input
    food_simulation.input["weight"] = weight_input
    food_simulation.input["height"] = height_input

    try:
        # Perform the simulation
        food_simulation.compute()

        # Output the recommendations
        return {
            "protein": round(food_simulation.output["protein"], 2),
            "fat": round(food_simulation.output["fat"], 2),
            "carbohydrates": round(food_simulation.output["carbohydrates"], 2),
            "magnesium": round(food_simulation.output["magnesium"], 2),
            "sodium": round(food_simulation.output["sodium"], 2),
            "potassium": round(food_simulation.output["potassium"], 2),
            "saturated_fats": round(food_simulation.output["saturated_fats"], 2),
            "monounsaturated_fats": round(
                food_simulation.output["monounsaturated_fats"], 2
            ),
            "polyunsaturated_fats": round(
                food_simulation.output["polyunsaturated_fats"], 2
            ),
            "free_sugar": round(food_simulation.output["free_sugar"], 2),
            "starch": round(food_simulation.output["starch"], 2),
        }

        # Repeat for other nutrients...
    except ValueError as e:
        print("Error during computation: ", e)
