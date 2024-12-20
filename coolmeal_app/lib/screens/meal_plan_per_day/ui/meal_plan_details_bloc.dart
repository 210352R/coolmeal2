
import 'package:coolmeal/models/meal_plan.dart';
import 'package:coolmeal/repositories/meal_repository.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:equatable/equatable.dart';

abstract class MealPlanDetailsEvent extends Equatable {
  @override
  List<Object?> get props => [];
}

class FetchMealPlanDetails extends MealPlanDetailsEvent {
  final int index;

  FetchMealPlanDetails(this.index);

  @override
  List<Object?> get props => [index];
}

// States
abstract class MealPlanDetailsState extends Equatable {
  @override
  List<Object?> get props => [];

}

class MealPlanDetailsInitial extends MealPlanDetailsState {}

class MealPlanDetailsLoading extends MealPlanDetailsState {}

class MealPlanDetailsLoaded extends MealPlanDetailsState {
  final MealPlan mealPlan;

  MealPlanDetailsLoaded(this.mealPlan);

  @override
  List<Object?> get props => [mealPlan];
}

class MealPlanDetailsError extends MealPlanDetailsState {
  final String error;

  MealPlanDetailsError(this.error);

  @override
  List<Object?> get props => [error];
}

// Bloc
class MealPlanDetailsBloc
    extends Bloc<MealPlanDetailsEvent, MealPlanDetailsState> {
  final MealRepository mealRepository;

  MealPlanDetailsBloc(this.mealRepository) : super(MealPlanDetailsInitial()) {
    on<FetchMealPlanDetails>(_onFetchMealPlanDetails);
  }

  Future<void> _onFetchMealPlanDetails(
      FetchMealPlanDetails event, Emitter<MealPlanDetailsState> emit) async {
    emit(MealPlanDetailsLoading());
    try {
      
      final mealPlan = await mealRepository.getMealPlanByIndex(
          event.index);
      if (mealPlan == null) {
        emit(MealPlanDetailsError("No meal plan found"));
      } else {
        emit(MealPlanDetailsLoaded(mealPlan));
      }
    } catch (e) {
      emit(MealPlanDetailsError(e.toString()));
    }
  }
}
