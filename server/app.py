import os
import requests
import pandas as pd
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from schema.userRequest import UserRequest
from schema.meal_plan import MealPlan
from schema.meal import Meal
from schema.ingredient import Ingredient
from schema.meal_item import MealItem
from schema.meal_name import MealName
from fastapi import FastAPI, HTTPException
from real_fuzzy_logic import fuzzy_recommend_nutrients
from bmr import calculate_bmr, calculate_daily_calories
from model import train_knn_model
from model import predict_knn
from week_prediction import week_prediction
from admin_helper import add_new_meal_plan
from mongodb_helper import (
    get_meal_plan_by_index,
    serialize_meal_plan,
    get_all_meal_plans,
    get_meal_by_name,
    get_top_50_breakfast_meals,
    get_top_50_lunch_meals,
    get_top_50_dinner_meals,
    get_total_meal_plans_count,
    get_last_index,
    insert_new_meal_plan,
    insert_new_final_meal,
    read_all_ingredients,
    read_one_ingredient,
    insert_one_ingredient,
    update_one_ingredient,
    delete_one_ingredient,
    convert_to_dict,
    read_all_meal_items,
    read_one_meal_item,
    insert_one_meal_item,
    update_one_meal_item,
    delete_one_meal_item,
    read_all_meals,
    read_one_meal,
    insert_one_meal,
    update_one_meal,
    delete_one_meal,
    read_all_meal_plans,
    read_one_meal_plan,
    insert_one_meal_plan,
    update_one_meal_plan,
    delete_one_meal_plan,
)
from typing import List, Optional


if not os.path.exists("./FinalPermutations.csv"):
    s3_url = "https://coolmeal.s3.amazonaws.com/FinalPermutations.csv"
    print(f"File not found at ./FinalPermutations.csv. Downloading from S3...")
    response = requests.get(s3_url)

    if response.status_code == 200:
        with open("./FinalPermutations.csv", "wb") as f:
            f.write(response.content)
        print(
            f"File downloaded successfully from S3 and saved to ./FinalPermutations.csv."
        )
        csv_file_path = "./FinalPermutations.csv"
        print(f"Reading data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)

        df["index"] = range(len(df))
        df.to_csv(csv_file_path, index=False)
        print(f"Index column added and CSV file saved back to {csv_file_path}.")
    else:
        raise FileNotFoundError(
            f"Failed to download file from {s3_url}. HTTP status code: {response.status_code}"
        )


app = FastAPI()

df = pd.read_csv("./FinalPermutations.csv")
print("Data set read successfully --------------------- ")


model = train_knn_model()
print("Model trained successfully --------------------- ")


@app.get("/read")
def read_root():
    return {"Hello": "World"}


@app.post(
    "/prediction", response_description="Get prediction for a user", status_code=200
)
def read_prediction(request: UserRequest):
    meal_plans = []
    nut_result = fuzzy_recommend_nutrients(
        request.age,
        request.weight,
        request.height,
        request.diabetes_input,
        request.pressure_input,
        request.chol_input,
    )
    tot_bmr = calculate_bmr(request.weight, request.height, request.age, request.gender)
    tot_kalories = calculate_daily_calories(tot_bmr, request.activity_level)

    input_data = [
        [
            float(request.price),
            tot_kalories,
            nut_result["protein"],
            nut_result["fat"],
            nut_result["carbohydrates"],
            nut_result["magnesium"],
            nut_result["sodium"],
            nut_result["potassium"],
            nut_result["saturated_fats"],
            nut_result["monounsaturated_fats"],
            nut_result["polyunsaturated_fats"],
            nut_result["free_sugar"],
            nut_result["starch"],
        ]
    ]

    print(input_data)
    prediction = predict_knn("ml_model.pkl", input_data)
    print(prediction)
    output = df.iloc[prediction[0]].to_dict(orient="records")

    print(output[0])
    meal_plans.append(output[0])

    week_plan = week_prediction(
        df, float(request.price), nut_result, tot_kalories, output[0], meal_plans
    )
    return JSONResponse(status_code=200, content={"prediction": week_plan})


@app.get("/get-mealplan-index")
async def getMealPlanByIndex(index: int):
    try:
        meal_plan = await get_meal_plan_by_index(index)

        if not meal_plan:
            raise HTTPException(status_code=404, detail="Meal plan not found")

        serialized_meal_plan = serialize_meal_plan(meal_plan)

        return JSONResponse(
            status_code=200, content={"meal_plan": serialized_meal_plan}
        )

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"detail": "An error occurred", "error": str(e)}
        )


# Modified method -------------


@app.post("/meals/byname")
async def get_meal_by_name_api(mealName: MealName):
    try:
        meal = await get_meal_by_name(mealName.name)

        if not meal:
            raise HTTPException(status_code=404, detail="Meal plan not found")

        serialized_meal_plan = serialize_meal_plan(meal)

        return JSONResponse(
            status_code=200, content={"meal_plan": serialized_meal_plan}
        )

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"detail": "An error occurred", "error": str(e)}
        )


@app.get(
    "/meals/breakfast/top50",
    response_description="Get top 50 meals for breakfast",
    status_code=200,
)
async def get_top_50_breakfast_meals_api():
    try:
        meals = await get_top_50_breakfast_meals()

        if not meals:
            raise HTTPException(status_code=404, detail="Meals not found")
        serialized_meals = [serialize_meal_plan(meal) for meal in meals]

        return JSONResponse(status_code=200, content={"meals": serialized_meals})

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"detail": "An error occurred", "error": str(e)}
        )


@app.get(
    "/meals/lunch/top50",
    response_description="Get top 50 meals for lunch",
    status_code=200,
)
async def get_top_50_lunch_meals_api():
    try:
        meals = await get_top_50_lunch_meals()

        if not meals:
            raise HTTPException(status_code=404, detail="Meals not found")

        serialized_meals = [serialize_meal_plan(meal) for meal in meals]

        return JSONResponse(status_code=200, content={"meals": serialized_meals})

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"detail": "An error occurred", "error": str(e)}
        )


@app.get(
    "/meals/dinner/top50",
    response_description="Get top 50 meals for dinner",
    status_code=200,
)
async def get_top_50_dinner_meals_api():
    try:
        meals = await get_top_50_dinner_meals()

        if not meals:
            raise HTTPException(status_code=404, detail="Meals not found")
        serialized_meals = [serialize_meal_plan(meal) for meal in meals]

        return JSONResponse(status_code=200, content={"meals": serialized_meals})

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"detail": "An error occurred", "error": str(e)}
        )


@app.get("/mealplans/count")
async def get_total_meal_plans_count_api():
    try:

        count = await get_total_meal_plans_count()
        return JSONResponse(status_code=200, content={"count": count})

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"detail": "An error occurred", "error": str(e)}
        )


# create method for last meal plan index
@app.get("/mealplans/last")
async def get_last_meal_plan_api():
    try:
        last_index = await get_last_index()
        return JSONResponse(status_code=200, content={"last_index": last_index})

    except Exception as e:
        # Catch any exceptions and return a 500 response
        return JSONResponse(
            status_code=500, content={"detail": "An error occurred", "error": str(e)}
        )


@app.post("/mealplans", response_description="Add new meal plan", status_code=201)
async def create_meal_plan(meal_plan: MealPlan):
    try:
        meal_plan_dict = meal_plan.dict()

        await add_new_meal_plan(meal_plan_dict)
        return JSONResponse(status_code=201, content={"data": "Success"})

    except ValueError as ve:
        #
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),  # Include the error message
        )
    except Exception as e:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while creating the meal plan.",  # General error message
        )


# Read all ingredients
@app.get("/ingredients", response_model=List[Ingredient])
def read_ingredients():
    ingredients = read_all_ingredients()
    return ingredients


# Read an ingredient by Food_Code
@app.get("/ingredients/{food_code}", response_model=Ingredient)
def read_ingredient(food_code: str):
    ingredient = read_one_ingredient(food_code)
    if not ingredient:
        raise HTTPException(status_code=404, detail="Ingredient not found")
    return convert_to_dict(ingredient)


# Insert a new ingredient
@app.post("/ingredients", response_model=Ingredient)
def create_ingredient(ingredient: Ingredient):
    if read_one_ingredient(ingredient.Food_Code):
        raise HTTPException(
            status_code=400, detail="Ingredient with this Food_Code already exists"
        )
    ingredient_data = ingredient.dict(by_alias=True)
    result = insert_one_ingredient(ingredient_data)
    ingredient_data["_id"] = str(result.inserted_id)
    return ingredient_data


# Update an ingredient by Food_Code
@app.put("/ingredients/{food_code}", response_model=Ingredient)
def update_ingredient(food_code: str, updated_ingredient: Ingredient):
    if not read_one_ingredient(food_code):
        raise HTTPException(status_code=404, detail="Ingredient not found")

    update_data = updated_ingredient.dict(by_alias=True)
    update_one_ingredient(food_code, update_data)
    updated_ingredient = read_one_ingredient(food_code)
    return convert_to_dict(updated_ingredient)


# Delete an ingredient by Food_Code
@app.delete("/ingredients/{food_code}")
def delete_ingredient(food_code: str):
    result = delete_one_ingredient(food_code)
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Ingredient not found")
    return {"message": "Ingredient deleted successfully"}


# Read all ingredients
@app.get("/mealitems", response_model=List[MealItem])
def read_ingredients():
    mealitems = read_all_meal_items()
    return mealitems


# Read an meal_item by Food_Code
@app.get("/mealitems/{id}", response_model=MealItem)
def read_meal_item(id: str):
    meal_item = read_one_meal_item(id)
    if not meal_item:
        raise HTTPException(status_code=404, detail="Meal item not found")
    return convert_to_dict(meal_item)


# Insert a new meal_item
@app.post("/mealitems", response_model=MealItem)
def create_meal_item(meal_item: MealItem):
    if read_one_meal(meal_item.Food_Code):
        raise HTTPException(
            status_code=400, detail="Meal item with this Food_Code already exists"
        )
    meal_item_data = meal_item.dict(by_alias=True)
    result = insert_one_meal_item(meal_item_data)
    meal_item_data["_id"] = str(result.inserted_id)
    return meal_item_data


# Update an mealitems by Food_Code
@app.put("/mealitems/{id}", response_model=MealItem)
def update_meal_item(id: str, updated_meal_item: MealItem):
    if not read_one_meal(id):
        raise HTTPException(status_code=404, detail="Meal item not found")

    update_data = updated_meal_item.dict(by_alias=True)
    update_one_meal_item(id, update_data)
    updated_meal_item = read_one_meal_item(id)
    return convert_to_dict(updated_meal_item)


# Delete an meal_item by Food_Code
@app.delete("/mealitems/{id}")
def delete_meal_item(food_code: str):
    result = delete_one_meal_item(id)
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Meal item not found")
    return {"message": "Meal item deleted successfully"}


# Read all meals
@app.get("/meals", response_model=List[Meal])
def read_ingredients():
    meals = read_all_meals()
    return meals


# Read an meal by Food_Code
@app.get("/meals/{id}", response_model=Meal)
def read_meal(id: str):
    meal = read_one_meal(id)
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")
    return convert_to_dict(meal)


@app.post("/meals", response_description="Add new meal", status_code=201)
async def add_meal(meal: Meal):
    try:
        meal_data = meal.dict()
        new_meal = await insert_new_final_meal(meal_data)

        return {"status": "success", "data": str(new_meal["_id"])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Update an meals by Food_Code
@app.put("/meals/{id}", response_model=Meal)
def update_meal_item(id: str, updated_meal: Meal):
    if not read_one_meal(id):
        raise HTTPException(status_code=404, detail="Meal not found")

    update_data = updated_meal.dict(by_alias=True)
    update_one_meal(id, update_data)
    updated_meal = read_one_meal(id)
    return convert_to_dict(updated_meal)


# Delete an meals
@app.delete("/meals/{id}")
def delete_meal(food_code: str):
    result = delete_one_meal(id)
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Meal not found")
    return {"message": "Meal deleted successfully"}


# Read all meal plans
@app.get("/mealplans", response_model=List[MealPlan])
def read_ingredients():
    m = read_all_meal_plans()
    return m


# Read an meal plan
@app.get("/mealplans/{id}", response_model=MealPlan)
def read_meal_plan(id: str):
    meal = read_one_meal_plan(id)
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")
    return convert_to_dict(meal)


@app.post("/mealplans", response_description="Add new meal plan", status_code=201)
async def add_meal(mealPlan: MealPlan):
    try:
        md = mealPlan.dict()
        new_meal = await insert_one_meal_plan(md)

        return {"status": "success", "data": str(new_meal["_id"])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Update an meals by Food_Code
@app.put("/mealplans/{id}", response_model=Meal)
def update_meal_plan(id: str, updated_meal_plan: Meal):
    if not read_one_meal(id):
        raise HTTPException(status_code=404, detail="Meal plan not found")

    update_data = updated_meal_plan.dict(by_alias=True)
    update_one_meal_plan(id, update_data)
    updated_meal_plan = read_one_meal_plan(id)
    return convert_to_dict(updated_meal_plan)


# Delete an meals
@app.delete("/mealplans/{id}")
def delete_meal(id: str):
    result = delete_one_meal_plan(id)
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    return {"message": "Meal deleted successfully"}


print("App is running on port 8000 ****************************************")
