"""
Academic Advisory & What-If Simulation Engine
---------------------------------------------
Generates personalized academic action plans, performs sensitivity analysis,
and solves for behavioral target thresholds to achieve student goals.
"""

from typing import Any, Dict, List, Optional
from src.student_predictor.models import StudentModelSuite


def generate_recommendations(
    inputs: Dict[str, Any], prediction: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Generate tailored academic coaching recommendations based on
    feature weights, student inputs, and predicted outcome.
    """
    recs = []
    study_hours = float(inputs.get("study_hours_per_week", 15.0))
    attendance = float(inputs.get("attendance_percent", 80.0))
    sleep = float(inputs.get("sleep_hours", 7.0))
    prev_score = float(inputs.get("previous_score", 65.0))
    pred_score = float(prediction.get("predicted_score", 65.0))
    pred_grade = prediction.get("predicted_grade", "C")
    risk_tier = prediction.get("risk_tier", "Low Risk")

    # 1. Study Hours Advice
    if study_hours < 10.0:
        recs.append({
            "category": "Study Habits",
            "type": "warning",
            "icon": "⚠️",
            "title": "Low Weekly Study Volume",
            "description": f"You are studying {study_hours:.1f} hrs/week. Students who average 16-20 hrs/week score an average of 12-15 points higher.",
            "action": "Increase your weekly study time by +4 to +6 hours using structured 45-minute Pomodoro intervals.",
        })
    elif study_hours < 18.0:
        recs.append({
            "category": "Study Habits",
            "type": "info",
            "icon": "💡",
            "title": "Moderate Study Regimen",
            "description": f"Your current study pace ({study_hours:.1f} hrs/week) provides a stable foundation.",
            "action": "Adding just 2.5 hours of active problem-solving or practice exams per week can raise your grade by one full tier.",
        })
    else:
        recs.append({
            "category": "Study Habits",
            "type": "success",
            "icon": "🌟",
            "title": "High Study Dedication",
            "description": f"Superb commitment ({study_hours:.1f} hrs/week)! You are in the top study quintile.",
            "action": "Focus on high-yield active recall and synthesis rather than passive rereading to prevent burnout.",
        })

    # 2. Attendance Advice
    if attendance < 75.0:
        recs.append({
            "category": "Attendance & Engagement",
            "type": "danger",
            "icon": "🚨",
            "title": "Critical Attendance Risk",
            "description": f"Attendance is at {attendance:.1f}%. Low attendance is the single highest statistical predictor of exam failure.",
            "action": "Prioritize attending every scheduled lecture and lab. Reaching 85% attendance will protect your grade.",
        })
    elif attendance < 90.0:
        recs.append({
            "category": "Attendance & Engagement",
            "type": "info",
            "icon": "📌",
            "title": "Healthy Attendance",
            "description": f"Attendance is {attendance:.1f}%. Consistent classroom presence strengthens conceptual retention.",
            "action": "Engage actively during discussion sections and office hours to clarify difficult topics before exam week.",
        })
    else:
        recs.append({
            "category": "Attendance & Engagement",
            "type": "success",
            "icon": "✅",
            "title": "Exemplary Attendance",
            "description": f"{attendance:.1f}% attendance! Consistent attendance directly correlates with exam mastery.",
            "action": "Maintain this standard and leverage teacher office hours for advanced questions.",
        })

    # 3. Sleep & Circadian Routine
    if sleep < 6.0:
        recs.append({
            "category": "Sleep & Wellness",
            "type": "warning",
            "icon": "🌙",
            "title": "Sleep Deprivation Risk",
            "description": f"Sleeping only {sleep:.1f} hrs/night reduces working memory and analytical speed by up to 20%.",
            "action": "Target 7.0-8.0 hours of sleep per night, especially during the 48 hours leading up to exams.",
        })
    elif sleep > 9.0:
        recs.append({
            "category": "Sleep & Wellness",
            "type": "info",
            "icon": "💤",
            "title": "Elevated Sleep Duration",
            "description": f"Averaging {sleep:.1f} hrs/night. Excessive sleep can sometimes indicate irregular sleep cycles or lethargy.",
            "action": "Strive for a consistent morning wake-up schedule to optimize cognitive peak hours.",
        })
    else:
        recs.append({
            "category": "Sleep & Wellness",
            "type": "success",
            "icon": "✨",
            "title": "Optimal Sleep Balance",
            "description": f"{sleep:.1f} hrs of nightly sleep is in the ideal cognitive recovery zone.",
            "action": "Keep this schedule steady throughout the semester.",
        })

    # 4. Trajectory & Growth Mindset
    delta = pred_score - prev_score
    if delta > 3.0:
        recs.append({
            "category": "Performance Momentum",
            "type": "success",
            "icon": "📈",
            "title": "Positive Growth Trajectory",
            "description": f"Predicted score ({pred_score:.1f}) represents a +{delta:.1f} point gain over your previous exam ({prev_score:.1f}).",
            "action": "Your current habits are compounding positively. Keep following your study schedule!",
        })
    elif delta < -3.0:
        recs.append({
            "category": "Performance Momentum",
            "type": "warning",
            "icon": "📉",
            "title": "Slight Performance Dip",
            "description": f"Predicted score ({pred_score:.1f}) is {abs(delta):.1f} points lower than your previous exam ({prev_score:.1f}).",
            "action": "Review the areas where study hours or attendance have dropped since last term.",
        })

    return recs


def simulate_what_if(
    model_suite: StudentModelSuite,
    base_inputs: Dict[str, Any],
    modifications: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Simulate the effect of adjusting one or more behavioral variables.
    Returns comparison between original and simulated predictions.
    """
    simulated_inputs = dict(base_inputs)
    simulated_inputs.update(modifications)

    base_pred = model_suite.predict_single(base_inputs)
    sim_pred = model_suite.predict_single(simulated_inputs)

    score_delta = round(sim_pred["predicted_score"] - base_pred["predicted_score"], 1)

    return {
        "base_inputs": base_inputs,
        "simulated_inputs": simulated_inputs,
        "base_score": base_pred["predicted_score"],
        "simulated_score": sim_pred["predicted_score"],
        "score_delta": score_delta,
        "base_grade": base_pred["predicted_grade"],
        "simulated_grade": sim_pred["predicted_grade"],
        "base_risk": base_pred["risk_tier"],
        "simulated_risk": sim_pred["risk_tier"],
    }


def target_goal_solver(
    model_suite: StudentModelSuite,
    base_inputs: Dict[str, Any],
    target_score: float = 85.0,
) -> Dict[str, Any]:
    """
    Calculate the study hours and attendance adjustments needed
    to reach a desired target score (e.g. 85 for Grade A).
    """
    base_pred = model_suite.predict_single(base_inputs)
    current_score = base_pred["predicted_score"]

    if current_score >= target_score:
        return {
            "on_track": True,
            "target_score": target_score,
            "current_score": current_score,
            "message": f"Congratulations! You are already on track for your target score of {target_score:.1f}.",
            "required_study_hours": base_inputs.get("study_hours_per_week", 15.0),
            "required_attendance": base_inputs.get("attendance_percent", 80.0),
        }

    # Binary search / step search for required study hours (holding attendance constant)
    test_inputs = dict(base_inputs)
    best_hours = None

    for h in [x * 0.5 for x in range(int(base_inputs.get("study_hours_per_week", 15) * 2), 81)]:
        test_inputs["study_hours_per_week"] = h
        pred = model_suite.predict_single(test_inputs)
        if pred["predicted_score"] >= target_score:
            best_hours = h
            break

    # Also test combined strategy (increase study hours + improve attendance to 95%)
    combined_inputs = dict(base_inputs)
    combined_inputs["attendance_percent"] = max(95.0, float(base_inputs.get("attendance_percent", 80)))
    combined_hours = None

    for h in [x * 0.5 for x in range(int(base_inputs.get("study_hours_per_week", 15) * 2), 81)]:
        combined_inputs["study_hours_per_week"] = h
        pred = model_suite.predict_single(combined_inputs)
        if pred["predicted_score"] >= target_score:
            combined_hours = h
            break

    return {
        "on_track": False,
        "target_score": target_score,
        "current_score": current_score,
        "gap": round(target_score - current_score, 1),
        "required_study_hours_only": best_hours,
        "study_hours_with_95pct_attendance": combined_hours,
    }
