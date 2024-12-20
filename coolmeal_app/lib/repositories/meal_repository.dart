import 'dart:convert';
import 'package:coolmeal/main.dart';
import 'package:http/http.dart' as http;
import 'package:coolmeal/models/meal.dart';
import 'package:coolmeal/models/meal_plan.dart';

class MealRepository {
  MealRepository();

  Future<Meal> getMealById(String mealId) async {
    final url = Uri.parse('$ServerIP/meals/$mealId');
    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        return Meal.fromJson(data);
      } else {
        throw Exception(
            'Failed to load meal. Status code: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error fetching meal: $e');
    }
  }

  Future<Meal> getMealByName(String mealName) async {
    final url = Uri.parse('$ServerIP/meals/byname');

    try {
      final response = await http.post(url, 
      headers: {'Content-Type': 'application/json'},body: jsonEncode({'name': mealName}));
      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;

        // TODO: CHANGE THIS BELOW LINE LATER
        return Meal.fromJson(data["meal_plan"]);
      } else {
        throw  Exception(
            'Failed to load meal. Status code: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error fetching meal: $e');
    }
  }

  Future<MealPlan?> getMealPlanByMealNames(
      String breakfast, String lunch, String dinner) async {
    final url = Uri.parse(
        '$ServerIP/mealplan?breakfast=$breakfast&lunch=$lunch&dinner=$dinner');
    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        return MealPlan.fromJson(data);
      } else {
        throw Exception(
            'Failed to load meal plan. Status code: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error fetching meal plan: $e');
    }
  }

  Future<MealPlan?> getMealPlanByIndex(int index) async {
    final url = Uri.parse('$ServerIP/get-mealplan-index?index=$index');
    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        return MealPlan.fromJson(data["meal_plan"]);
      } else {
        throw Exception(
            'Failed to load meal plan. Status code: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error fetching meal plan: $e');
    }
  }
}
