import cv2
import mediapipe as mp

def analyze_eye_contact(video_path):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    # Helper functions
    def get_point(landmarks, index, width, height):
        lm = landmarks[index]
        return int(lm.x * width), int(lm.y * height)

    def get_vertical_ratio(iris, upper, lower):
        if lower[1] == upper[1]:
            return 0.5
        return (iris[1] - upper[1]) / (lower[1] - upper[1])

    def get_horizontal_ratio(iris, outer, inner):
        if inner[0] == outer[0]:
            return 0.5
        return (iris[0] - outer[0]) / (inner[0] - outer[0])

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    looking_away_frames = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        gaze_text = "Looking away"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                lm = face_landmarks.landmark

                try:
                    # Iris and eye reference points
                    left_iris = get_point(lm, 468, w, h)
                    left_outer = get_point(lm, 33, w, h)
                    left_inner = get_point(lm, 133, w, h)
                    left_upper = get_point(lm, 159, w, h)
                    left_lower = get_point(lm, 145, w, h)

                    right_iris = get_point(lm, 473, w, h)
                    right_outer = get_point(lm, 362, w, h)
                    right_inner = get_point(lm, 263, w, h)
                    right_upper = get_point(lm, 386, w, h)
                    right_lower = get_point(lm, 374, w, h)

                    # Skip closed eyes
                    min_eye_open = 5
                    if abs(left_upper[1] - left_lower[1]) < min_eye_open or abs(right_upper[1] - right_lower[1]) < min_eye_open:
                        raise ValueError("Eyes likely closed")

                    # Gaze ratios
                    left_h_ratio = get_horizontal_ratio(left_iris, left_outer, left_inner)
                    right_h_ratio = get_horizontal_ratio(right_iris, right_outer, right_inner)
                    avg_h_ratio = (left_h_ratio + right_h_ratio) / 2

                    left_v_ratio = get_vertical_ratio(left_iris, left_upper, left_lower)
                    right_v_ratio = get_vertical_ratio(right_iris, right_upper, right_lower)
                    avg_v_ratio = (left_v_ratio + right_v_ratio) / 2

                    # Additional face info
                    iris_y_offset = (lm[468].y + lm[473].y)/2 - (lm[159].y + lm[386].y + lm[145].y + lm[374].y)/4
                    normed_iris_y = (lm[468].y + lm[473].y) / 2
                    pitch = (lm[10].y - lm[152].y) if 10 in range(len(lm)) and 152 in range(len(lm)) else 0.25

                    # Gaze classification
                    if avg_h_ratio < 0.35 or avg_h_ratio > 0.65:
                        gaze_text = "Looking away"
                    else:
                        if avg_v_ratio < 0.44:
                            if iris_y_offset < -0.0025 and normed_iris_y < 0.428 and pitch > 0.238:
                                gaze_text = "Looking away"
                            elif -0.0020 <= iris_y_offset <= -0.0010 and normed_iris_y >= 0.435 and pitch < 0.234:
                                gaze_text = "Looking at Camera"
                            else:
                                gaze_text = "Looking away"
                        elif avg_v_ratio > 0.75:
                            gaze_text = "Looking away"
                        else:
                            gaze_text = "Looking at Camera"

                except:
                    gaze_text = "Looking away"
        else:
            gaze_text = "Looking away"

        if gaze_text == "Looking away":
            looking_away_frames += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if total_frames > 0:
        contact_loss = 0.8 * (looking_away_frames / total_frames)
        return f"{contact_loss * 100:.2f}% contact loss"
    else:
        return "0.00% contact loss"


# if __name__ == "__main__":
#     video_path = "eye_contact_analysis/lookingdown.mp4"  # Replace with your video path
#     result = analyze_eye_contact(video_path)
#     print(result)