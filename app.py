from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)

# =========================
# SUBJECTS
# =========================
subjects = {
    "DSA": ["Arrays", "Linked List", "Trees", "Graphs", "Hashing"],
    "Math": ["Algebra", "Calculus", "Probability"]
}

# =========================
# TRAIN ML MODEL
# =========================
def train_model():
    scores = np.array([[30], [40], [50], [60], [70], [80], [90]])
    study_time = np.array([3, 3, 2.5, 2, 1.5, 1, 0.5])

    model = LinearRegression()
    model.fit(scores, study_time)
    return model

model = train_model()

def predict_study_time(score):
    return model.predict([[score]])[0]

# =========================
# IMAGE DETECTION
# =========================
def detect_free_slots(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    height, width = thresh.shape

    rows = 5
    cols = 6

    cell_h = height // rows
    cell_w = width // cols

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    free_slots = []

    for i in range(rows):
        for j in range(cols):
            cell = thresh[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]

            white_pixels = np.sum(cell == 255)
            total_pixels = cell.size

            if white_pixels / total_pixels > 0.8:
                free_slots.append((days[i], f"Slot {j+1}"))

    return free_slots

# =========================
# HOME
# =========================
@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# =========================
# MAIN API
# =========================
@app.route('/generate-plan', methods=['POST'])
def generate_plan():
    scores = eval(request.form.get("scores"))

    file = request.files['image']
    image_path = "uploaded.png"
    file.save(image_path)

    free_slots = detect_free_slots(image_path)

    subject_priority = {
        sub: predict_study_time(score) for sub, score in scores.items()
    }

    sorted_subjects = sorted(subject_priority.items(), key=lambda x: x[1], reverse=True)

    schedule = []
    topic_index = {sub: 0 for sub in subjects}

    for i, (day, slot) in enumerate(free_slots):
        subject = sorted_subjects[i % len(sorted_subjects)][0]

        topic = subjects[subject][topic_index[subject] % len(subjects[subject])]
        topic_index[subject] += 1

        schedule.append({
            "day": day,
            "slot": slot,
            "task": f"{subject} - {topic}"
        })

    return jsonify(schedule)

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)