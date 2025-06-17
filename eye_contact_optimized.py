import cv2
import mediapipe as mp

def get_point(landmarks, idx, width, height):
    lm = landmarks[idx]
    return int(lm.x * width), int(lm.y * height)

def is_aligned_h(p1, p2, center, tol=15):
    return abs(p1[1] - center[1]) < tol and abs(p2[1] - center[1]) < tol

def is_aligned_v(p1, p2, center, tol=15):
    return abs(p1[0] - center[0]) < tol and abs(p2[0] - center[0]) < tol

def is_iris_centered(iris, outer, inner, margin=15):
    eye_mid_x = (outer[0] + inner[0]) // 2
    return abs(iris[0] - eye_mid_x) < margin

def is_plus_strict(iris, up, down, outer, inner, margin=15):
    return (
        is_aligned_v(up, down, iris, margin) and
        is_aligned_h(outer, inner, iris, margin) and
        is_iris_centered(iris, outer, inner, margin)
    )

def calculate_eye_contact_loss(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Video couldn't be opened.")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    red_frames = 0
    total_frames = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        label = "Red"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark
                try:
                    # LEFT eye
                    l_center = get_point(lm, 468, w, h)
                    l_outer = get_point(lm, 33, w, h)
                    l_inner = get_point(lm, 133, w, h)
                    l_upper = get_point(lm, 159, w, h)
                    l_lower = get_point(lm, 145, w, h)

                    # RIGHT eye
                    r_center = get_point(lm, 473, w, h)
                    r_outer = get_point(lm, 362, w, h)
                    r_inner = get_point(lm, 263, w, h)
                    r_upper = get_point(lm, 386, w, h)
                    r_lower = get_point(lm, 374, w, h)

                    # Check alignment
                    left_plus = is_plus_strict(l_center, l_upper, l_lower, l_outer, l_inner)
                    right_plus = is_plus_strict(r_center, r_upper, r_lower, r_outer, r_inner)

                    if left_plus and right_plus:
                        label = "Green"
                except:
                    label = "Red"

        if label == "Red":
            red_frames += 1
        total_frames += 1

    cap.release()

    if total_frames > 0:
        eye_contact_loss = (red_frames / total_frames) * 100
        print(f"{eye_contact_loss:.2f}% eye contact loss")
    else:
        print("No frames processed.")

# -----------------------------
# Run the function with video input
# -----------------------------
# if __name__ == "__main__":
#     video_file = 'closed-eyes.mp4'  # Replace with your input file
#     calculate_eye_contact_loss(video_file)
