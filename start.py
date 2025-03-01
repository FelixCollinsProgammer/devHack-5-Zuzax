import sys
import cv2
import mediapipe as mp
import random
import math
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QStackedWidget,
                             QGraphicsDropShadowEffect, QMessageBox, QFrame, QLineEdit)
from PyQt5.QtGui import (QImage, QPixmap, QFont, QIcon, QColor, QLinearGradient,
                         QPainter, QBrush, QPen, QRadialGradient)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve


class HandTrackingThread(QThread):
    image_data = pyqtSignal(QImage)
    gesture_detected = pyqtSignal(str, float, float)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cap = None
        self._run_flag = True
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.prev_gesture = None
        self.gesture_buffer = []
        self.buffer_size = 3  # Number of frames for gesture smoothing

    def run(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.error_signal.emit("Could not open camera!")
                return

            while self._run_flag:
                ret, frame = self.cap.read()
                if not ret:
                    self.error_signal.emit("Error reading frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                                    landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                                    connection_drawing_spec=self.mp_draw.DrawingSpec(color=(128, 128, 128), thickness=1))

                        gesture = self.detect_gesture(hand_landmarks)

                        if hand_landmarks.landmark:
                            # Use wrist position for x, y coordinates (more stable)
                            x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
                            y = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y

                            if gesture:
                                # Gesture smoothing using a buffer
                                self.gesture_buffer.append(gesture)
                                if len(self.gesture_buffer) >= self.buffer_size:
                                    # Find the most frequent gesture in the buffer
                                    most_frequent_gesture = max(set(self.gesture_buffer), key=self.gesture_buffer.count, default=None)

                                    # Only emit the gesture if it has changed
                                    if most_frequent_gesture != self.prev_gesture:
                                        self.gesture_detected.emit(most_frequent_gesture, x, y)
                                        self.prev_gesture = most_frequent_gesture  # Update previous gesture
                                    self.gesture_buffer = [] # Clear buffer
                            else:
                                # No gesture detected: reset buffer and previous gesture
                                self.prev_gesture = None
                                self.gesture_buffer = []



                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                self.image_data.emit(q_img)

            self.cap.release()
        except Exception as e:
            error_message = f"Error in hand tracking thread: {str(e)}"  # More descriptive error
            self.error_signal.emit(error_message)

    def calculate_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def detect_gesture(self, landmarks):
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        thumb_mcp = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        index_mcp = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        # Calculate distances for gesture recognition
        thumb_index_dist = self.calculate_distance(thumb_tip, index_mcp)
        thumb_middle_dist = self.calculate_distance(thumb_tip, middle_mcp)
        index_middle_dist = self.calculate_distance(index_tip, middle_tip)


        # Check finger extension based on y-coordinate relative to MCP and wrist
        index_finger_extended = index_tip.y < index_mcp.y and index_tip.y < wrist.y
        middle_finger_extended = middle_tip.y < middle_mcp.y and middle_tip.y < wrist.y
        ring_finger_extended = ring_tip.y < ring_mcp.y and ring_tip.y < wrist.y
        pinky_finger_extended = pinky_tip.y < pinky_mcp.y and pinky_tip.y < wrist.y

        # Thumb extension is more complex due to its movement range
        thumb_extended = thumb_tip.x > thumb_mcp.x if thumb_tip.x > thumb_mcp.x else thumb_tip.x < wrist.x

        # --- Improved Gesture Recognition Logic ---

        # Rock: All fingers curled
        thumb_curl_Rock = thumb_tip.y > thumb_mcp.y
        index_curl_Rock = index_tip.y > index_mcp.y
        middle_curl_Rock = middle_tip.y > middle_mcp.y
        ring_curl_Rock = ring_tip.y > ring_mcp.y
        pinky_curl_Rock = pinky_tip.y > pinky_mcp.y


        if thumb_curl_Rock and index_curl_Rock and middle_curl_Rock and ring_curl_Rock and pinky_curl_Rock:
            return "Rock"

        # Paper: All fingers extended and thumb extended and up.
        elif index_finger_extended and middle_finger_extended and ring_finger_extended and pinky_finger_extended and thumb_extended and thumb_tip.y < wrist.y :
            return "Paper"

        # Scissors: Index and middle fingers extended, others curled, and sufficient distance between them.
        elif index_finger_extended and middle_finger_extended and not ring_finger_extended and not pinky_finger_extended and index_middle_dist > 0.08:
             return "Scissors"

        else:
            return None  # Gesture not recognized

    def stop(self):
        self._run_flag = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.wait()


class RPSResultFrame(QFrame):
    def __init__(self, player_choice, computer_choice, result_text):
        super().__init__()
        self.player_choice = player_choice
        self.computer_choice = computer_choice
        self.result_text = result_text
        self.setFixedSize(200, 160)
        self.setStyleSheet("""
            QFrame {
                border: 3px solid #008080;
                border-radius: 15px;
                background-color: rgba(0, 128, 128, 40);
            }
        """)
        self.initUI()
        self.hover_animation = QPropertyAnimation(self, b"geometry")
        self.hover_animation.setDuration(200)
        self.original_geometry = None

    def initUI(self):
        layout = QVBoxLayout()
        player_image_path = f"{self.player_choice.lower()}.png" if self.player_choice else "question.png"
        computer_image_path = f"{self.computer_choice.lower()}.png" if self.computer_choice else "question.png"
        player_pixmap = QPixmap(player_image_path).scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        computer_pixmap = QPixmap(computer_image_path).scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        player_label = QLabel()
        player_label.setPixmap(player_pixmap)
        player_label.setAlignment(Qt.AlignCenter)

        computer_label = QLabel()
        computer_label.setPixmap(computer_pixmap)
        computer_label.setAlignment(Qt.AlignCenter)

        result_label = QLabel(self.result_text)
        result_label.setAlignment(Qt.AlignCenter)
        font = QFont("Segoe UI", 14, QFont.Bold)
        result_label.setFont(font)

        layout.addWidget(player_label)
        layout.addWidget(computer_label)
        layout.addWidget(result_label)
        layout.setSpacing(8)
        self.setLayout(layout)

    def enterEvent(self, event):
        if self.original_geometry is None:
            self.original_geometry = self.geometry()
        self.hover_animation.setStartValue(self.geometry())
        self.hover_animation.setEndValue(self.original_geometry.adjusted(-5, -5, 5, 5))
        self.hover_animation.start()

    def leaveEvent(self, event):
        self.hover_animation.setStartValue(self.geometry())
        self.hover_animation.setEndValue(self.original_geometry)
        self.hover_animation.start()


class RockPaperScissorsGame(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Камень, Ножницы, Бумага")
        self.setMinimumSize(1200, 720)
        self.setWindowIcon(QIcon("icon.png"))
        self.player_name = ""
        self.player_score = 0
        self.computer_score = 0
        self.attempts = 0
        self.MAX_ATTEMPTS = 5
        self.results = []
        self.player_choice = None
        self.computer_choice = None
        self.choices = ["Rock", "Scissors", "Paper"]
        self.gesture_locked = False
        self.round_winner = None

        self.hand_tracking_thread = None
        self.create_thread()

        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_number = 3

        self.delay_timer = QTimer(self)
        self.delay_timer.setInterval(2200)
        self.delay_timer.timeout.connect(self.prepare_next_round)
        self.delay_timer.setSingleShot(True)

        self.stacked_widget = QStackedWidget()
        self.main_layout = QVBoxLayout(self)  # Main layout for the whole window
        self.main_layout.addWidget(self.stacked_widget)

        self.init_start_screen()
        self.init_game_screen()
        self.init_results_screen()

        self.stacked_widget.addWidget(self.start_screen)
        self.stacked_widget.addWidget(self.game_screen)
        self.stacked_widget.addWidget(self.results_screen)
        self.stacked_widget.setCurrentWidget(self.start_screen)

        self.setStyleSheet(self.get_stylesheet())

    def create_thread(self):
        if self.hand_tracking_thread:
            self.hand_tracking_thread.stop()
        self.hand_tracking_thread = HandTrackingThread()
        self.hand_tracking_thread.image_data.connect(self.update_image)
        self.hand_tracking_thread.gesture_detected.connect(self.handle_gesture)
        self.hand_tracking_thread.error_signal.connect(self.show_error)

    def init_start_screen(self):
        start_screen = QWidget()
        self.start_screen = start_screen
        layout = QVBoxLayout()

        layout.addStretch(2)

        title_label = QLabel("Камень, Ножницы, Бумага")
        title_label.setFont(QFont("Impact", 72, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #008080; margin-bottom: 0px;")
        self.animate_widget(title_label, duration=750, easing_curve=QEasingCurve.OutBounce)
        layout.addWidget(title_label)

        subtitle_label = QLabel("против Компьютера")
        subtitle_label.setFont(QFont("Arial", 28, QFont.Bold))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: white; margin-bottom: 20px;")
        layout.addWidget(subtitle_label)

        nickname_frame = QFrame()
        nickname_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 50);
                border-radius: 20px;
                padding: 15px;
            }
        """)
        nickname_layout = QHBoxLayout(nickname_frame)
        nickname_label = QLabel("Никнейм:")
        nickname_label.setFont(QFont("Arial", 18))
        nickname_label.setStyleSheet("color: white;")

        self.nickname_input = QLineEdit()
        self.nickname_input.setFont(QFont("Arial", 18))
        self.nickname_input.setStyleSheet("""
                QLineEdit {
                color: white;
                background-color: #333333;
                border: 2px solid #555555;
                border-radius: 15px;
                padding: 12px;
                selection-background-color: #008080;
                min-width: 250px;
            }
            QLineEdit:focus {
                border: 2px solid #008080;
            }
            """)
        self.nickname_input.setPlaceholderText("Введите ваш никнейм")
        self.nickname_input.textChanged.connect(self.check_nickname)

        nickname_layout.addWidget(nickname_label)
        nickname_layout.addWidget(self.nickname_input)
        layout.addWidget(nickname_frame, alignment=Qt.AlignCenter)

        self.start_button = QPushButton("Играть!")
        self.start_button.setFont(QFont("Arial", 28, QFont.Bold))
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #008080, stop: 1 #006666);
                color: white;
                border: none;
                padding: 18px 40px;
                border-radius: 25px;
                font-size: 28px;
                margin-top: 25px;

            }
            QPushButton:hover {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #006666, stop: 1 #004d4d);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                        stop: 0  #004d4d, stop: 1 #003333);
            }
            QPushButton:disabled {
               background-color: #666666;
                color: #999999;
            }
            """)
        self.start_button.clicked.connect(self.start_game)
        self.start_button.setEnabled(False)
        self.add_shadow(self.start_button)
        layout.addWidget(self.start_button, alignment=Qt.AlignCenter)

        rules_frame = QFrame()
        rules_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 50);
                border-radius: 20px;
                padding: 20px;
                margin-top: 30px;
            }
        """)

        rules_layout = QVBoxLayout(rules_frame)

        rules_text = (
            "<b>Как играть:</b><br><br>"
            "1. Покажите жестом Камень, Ножницы или Бумагу.<br>"
            "2. Игра начнет обратный отсчет с 3.<br>"
            "3. Покажите жест после окончания отсчета.<br><br>"
            "<b>Правила победы:</b><br><br>"
            "  - Камень ломает Ножницы<br>"
            "  - Ножницы режут Бумагу<br>"
            "  - Бумага накрывает Камень<br><br>"
            "<i>Побеждает лучший из 5 раундов!</i>"
        )
        rules_label = QLabel(rules_text)
        rules_label.setFont(QFont("Arial", 16))
        rules_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        rules_label.setWordWrap(True)
        rules_label.setStyleSheet("color: white;")
        rules_label.setContentsMargins(20, 10, 20, 20)

        rules_layout.addWidget(rules_label)
        layout.addWidget(rules_frame)

        layout.addStretch(2)
        start_screen.setLayout(layout)

    def check_nickname(self):
        if self.nickname_input.text().strip():
            self.start_button.setEnabled(True)
        else:
            self.start_button.setEnabled(False)

    def init_game_screen(self):
        game_screen = QWidget()
        self.game_screen = game_screen
        game_screen.setAutoFillBackground(True)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)

        header_layout = QVBoxLayout()

        game_title_label = QLabel("Игра Камень, Ножницы, Бумага")
        game_title_label.setFont(QFont("Impact", 52, QFont.Bold))
        game_title_label.setAlignment(Qt.AlignCenter)
        game_title_label.setStyleSheet("color: #008080;")
        header_layout.addWidget(game_title_label)

        instruction_label = QLabel("Покажите жест в камеру после начала отсчета")
        instruction_label.setFont(QFont("Arial", 16))
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("color: #AAAAAA;")
        header_layout.addWidget(instruction_label)

        # Player Section
        player_layout = QVBoxLayout()
        self.player_label = QLabel(f"Игрок: {self.player_name}")  # Display player's name
        self.player_label.setFont(QFont("Arial", 28, QFont.Bold))
        self.player_label.setAlignment(Qt.AlignCenter)
        self.player_label.setStyleSheet("color: #AAAAAA;")  # Light gray
        player_layout.addWidget(self.player_label)

        self.player_choice_label = QLabel()  # Displays player's choice image
        self.player_choice_label.setAlignment(Qt.AlignCenter)
        self.player_choice_label.setMinimumSize(120, 120)  # Ensure it's large enough
        player_layout.addWidget(self.player_choice_label)

        self.video_label = QLabel()  # Displays the camera feed
        self.video_label.setFixedSize(480, 360)
        self.video_label.setStyleSheet("""
                border: 3px solid;
                border-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #006666, stop: 1 #008080);
                border-radius: 15px;
                """)  # Rounded border with gradient
        player_layout.addWidget(self.video_label)
        player_layout.setAlignment(Qt.AlignCenter)

        # VS Label Section
        vs_layout = QVBoxLayout()
        vs_layout.addStretch()
        self.vs_label = QLabel(f"<font color='white'>V</font><font color='#008080'>S</font>")  # VS in different colors
        font = QFont("Impact", 120, QFont.Bold)  # Large, bold font
        self.vs_label.setFont(font)
        self.vs_label.setAlignment(Qt.AlignCenter)
        vs_layout.addWidget(self.vs_label)
        vs_layout.addStretch()

        # Computer Section
        computer_layout = QVBoxLayout()
        computer_label = QLabel("Компьютер")
        computer_label.setFont(QFont("Arial", 28, QFont.Bold))
        computer_label.setAlignment(Qt.AlignCenter)
        computer_label.setStyleSheet("color: #AAAAAA;")  # Light gray
        computer_layout.addWidget(computer_label)

        self.computer_choice_label = QLabel()  # Displays computer's choice image
        self.computer_choice_label.setAlignment(Qt.AlignCenter)
        self.computer_choice_label.setMinimumSize(120, 120)
        computer_layout.addWidget(self.computer_choice_label)

        # Label to show the computer's choice as text, below the image
        self.computer_choice_text_label = QLabel()
        self.computer_choice_text_label.setFixedSize(480, 360)
        self.computer_choice_text_label.setFont(QFont("Arial", 24))
        self.computer_choice_text_label.setAlignment(Qt.AlignCenter)
        self.computer_choice_text_label.setWordWrap(True)  # Wrap text if it's long
        self.computer_choice_text_label.setStyleSheet("""
                QLabel {
                    color: white;
                    font-size: 48px; /* Larger font size */
                    font-weight: bold;
                }
            """)
        computer_layout.addWidget(self.computer_choice_text_label)
        computer_layout.setAlignment(Qt.AlignCenter)

        main_horizontal_layout = QHBoxLayout()
        main_horizontal_layout.addLayout(player_layout)
        main_horizontal_layout.addStretch(1)  # Add stretch to center VS
        main_horizontal_layout.addLayout(vs_layout)
        main_horizontal_layout.addStretch(1)  # Add stretch
        main_horizontal_layout.addLayout(computer_layout)

        # Bottom Layout (Score, Attempts, Result, Countdown)
        bottom_layout = QVBoxLayout()

        self.score_label = QLabel("Игрок: 0   Компьютер: 0")  # Score display
        self.score_label.setFont(QFont("Arial", 22, QFont.Bold))
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setStyleSheet("color: white;")
        bottom_layout.addWidget(self.score_label)

        self.attempts_label = QLabel("Раунд: 1/5")  # Attempts display
        self.attempts_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.attempts_label.setAlignment(Qt.AlignCenter)
        self.attempts_label.setStyleSheet("color: white;")
        bottom_layout.addWidget(self.attempts_label)

        self.result_label = QLabel()  # Result of the round
        self.result_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: white;")
        bottom_layout.addWidget(self.result_label)

        self.countdown_label = QLabel()  # Countdown timer display
        self.countdown_label.setFont(QFont("Impact", 64, QFont.Bold))
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("color: #008080;")  # Teal color
        bottom_layout.addWidget(self.countdown_label)

        main_layout.addLayout(main_horizontal_layout)
        main_layout.addLayout(bottom_layout)
        game_screen.setLayout(main_layout)

    def init_results_screen(self):
        results_screen = QWidget()
        self.results_screen = results_screen
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Результаты")
        title_label.setFont(QFont("Impact", 48, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #008080;")
        layout.addWidget(title_label)

        # Container for round results
        self.rounds_results_container = QHBoxLayout()
        layout.addLayout(self.rounds_results_container)

        # Final Result Label
        self.final_result_label = QLabel()
        final_font = QFont("Arial", 36, QFont.Bold)  # Larger, bold font
        self.final_result_label.setFont(final_font)
        self.final_result_label.setAlignment(Qt.AlignCenter)
        self.final_result_label.setStyleSheet("color: white")
        layout.addWidget(self.final_result_label)

        # Restart Button
        restart_button = QPushButton("Переиграть")
        restart_button.setFont(QFont("Arial", 18, QFont.Bold))
        restart_button.clicked.connect(self.restart_game)
        self.add_shadow(restart_button)  # Add shadow for depth
        layout.addWidget(restart_button)

        results_screen.setLayout(layout)

    def init_results_screen(self):
        results_screen = QWidget()
        self.results_screen = results_screen
        layout = QVBoxLayout()
        layout.setSpacing(20)

        title_label = QLabel("Результаты")
        title_label.setFont(QFont("Impact", 56, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #008080; margin-bottom: 10px;")
        layout.addWidget(title_label)

        self.rounds_results_container = QHBoxLayout()

        scroll_area = QFrame()
        scroll_area.setLayout(self.rounds_results_container)
        scroll_area.setStyleSheet("background-color: transparent;")
        layout.addWidget(scroll_area)

        self.final_result_label = QLabel()
        final_font = QFont("Arial", 36, QFont.Bold)
        self.final_result_label.setFont(final_font)
        self.final_result_label.setAlignment(Qt.AlignCenter)
        self.final_result_label.setStyleSheet("""
            color: white;
            padding: 20px;
            border-radius: 15px;
            background-color: rgba(0, 0, 0, 50);
            margin-bottom: 10px;
            """)
        layout.addWidget(self.final_result_label)

        restart_button = QPushButton("Переиграть")
        restart_button.setFont(QFont("Arial", 20, QFont.Bold))
        restart_button.setStyleSheet("""
            QPushButton {
                 background-color: rgba(0, 128, 128, 80);
                color: white;
                border: 3px solid #008080;
                padding: 15px 32px;
                border-radius: 25px;
                font-size: 22px;

            }
            QPushButton:hover {
                 background-color: rgba(0, 102, 102, 80);
                 border-color: #006666;
            }
            QPushButton:pressed {
               background-color: rgba(0, 77, 77, 80);
            }
           """)
        restart_button.clicked.connect(self.restart_game)
        self.add_shadow(restart_button)
        layout.addWidget(restart_button, alignment=Qt.AlignCenter)

        results_screen.setLayout(layout)
        layout.addStretch()

    def add_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(5)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 180))
        widget.setGraphicsEffect(shadow)

    def animate_widget(self, widget, duration=500, start_value=0.0, end_value=1.0,
                       easing_curve=QEasingCurve.OutQuad):
        """Animates the opacity of a widget."""
        animation = QPropertyAnimation(widget, b"windowOpacity")
        animation.setDuration(duration)
        animation.setStartValue(start_value)
        animation.setEndValue(end_value)
        animation.setEasingCurve(easing_curve)
        animation.start()

    def animate_result(self):
        self.result_label.setWindowOpacity(0)
        animation = QPropertyAnimation(self.result_label, b"windowOpacity")
        animation.setDuration(500)
        animation.setStartValue(0)
        animation.setEndValue(1)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start()

    def start_game(self):
        self.player_name = self.nickname_input.text().strip()
        self.player_label.setText(f"Игрок: {self.player_name}")
        self.stacked_widget.setCurrentWidget(self.game_screen)
        self.reset_game_state()
        if not self.hand_tracking_thread.isRunning():
            self.hand_tracking_thread.start()
        self.start_countdown()

    def start_countdown(self):
        self.countdown_number = 3
        self.countdown_label.setText(str(self.countdown_number))
        self.countdown_timer.start()

    def update_countdown(self):
        self.countdown_number -= 1
        if self.countdown_number > 0:
            self.countdown_label.setText(str(self.countdown_number))
        else:
            self.countdown_label.setText("")
            self.countdown_timer.stop()
            self.computer_choice = random.choice(self.choices)

    def handle_gesture(self, gesture, x, y):
        if self.countdown_timer.isActive() or self.gesture_locked:
            return

        if self.computer_choice is not None:
            self.gesture_locked = True
            self.player_choice = gesture
            print(f"Detected gesture: {gesture}, x: {x}, y: {y}")
            self.display_player_choice()
            self.display_computer_choice()
            self.determine_winner()
            self.delay_timer.start()

    def prepare_next_round(self):
        if self.round_winner == "player":
            self.player_score += 1
        elif self.round_winner == "computer":
            self.computer_score += 1

        self.update_score()
        self.round_winner = None

        if self.attempts >= self.MAX_ATTEMPTS - 1:
            self.show_results()
        else:
            self.attempts += 1
            self.attempts_label.setText(
                f"Раунд: {self.attempts + 1}/{self.MAX_ATTEMPTS}")
            self.result_label.setText("")
            self.player_choice_label.clear()
            self.computer_choice_label.clear()
            self.computer_choice_text_label.setText("")
            self.player_choice = None
            self.computer_choice = None
            self.gesture_locked = False
            self.start_countdown()

    def determine_winner(self):
        if self.player_choice == self.computer_choice:
            result_text = "Ничья!"
            self.round_winner = "tie"
        elif (self.player_choice == "Rock" and self.computer_choice == "Scissors") or \
             (self.player_choice == "Scissors" and self.computer_choice == "Paper") or \
             (self.player_choice == "Paper" and self.computer_choice == "Rock"):
            result_text = "Игрок побеждает!"
            self.round_winner = "player"
        else:
            result_text = "Компьютер побеждает!"
            self.round_winner = "computer"

        self.result_label.setText(result_text)
        self.animate_result()
        self.results.append((self.player_choice, self.computer_choice, result_text))

    def update_score(self):
        self.score_label.setText(f"Игрок: {self.player_score}   Компьютер: {self.computer_score}")

    def display_player_choice(self):
        if self.player_choice:
            pixmap = QPixmap(f"{self.player_choice.lower()}.png")
            pixmap = pixmap.scaled(self.player_choice_label.width(), self.player_choice_label.height(), Qt.KeepAspectRatio)
            self.player_choice_label.setPixmap(pixmap)
        else:
            self.player_choice_label.clear()

    def display_computer_choice(self):
        if self.computer_choice:
            pixmap = QPixmap(f"{self.computer_choice.lower()}.png")
            pixmap = pixmap.scaled(self.computer_choice_label.width(), self.computer_choice_label.height(),
                                   Qt.KeepAspectRatio)
            self.computer_choice_label.setPixmap(pixmap)
            self.computer_choice_text_label.setText(
                self.computer_choice)
        else:
            self.computer_choice_label.clear()
            self.computer_choice_text_label.clear()

    def show_results(self):
        self.stacked_widget.setCurrentWidget(self.results_screen)
        for i in reversed(range(self.rounds_results_container.count())):
            widget = self.rounds_results_container.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        for player_choice, computer_choice, result_text in self.results:
            frame = RPSResultFrame(player_choice, computer_choice, result_text)
            self.rounds_results_container.addWidget(frame)

        if self.player_score > self.computer_score:
            self.final_result_label.setText(
                f"{self.player_name} побеждает в игре!")
        elif self.computer_score > self.player_score:
            self.final_result_label.setText("Компьютер побеждает в игре!")
        else:
            self.final_result_label.setText(
                "В игре ничья!")

    def reset_game_state(self):
        self.player_score = 0
        self.computer_score = 0
        self.attempts = 0
        self.results = []
        self.player_choice = None
        self.computer_choice = None
        self.gesture_locked = False
        self.round_winner = None

        self.update_score()
        self.attempts_label.setText("Раунд: 1/5")
        self.result_label.setText("")

    def restart_game(self):
        self.reset_game_state()
        self.create_thread()
        self.stacked_widget.setCurrentWidget(self.start_screen)

        for i in reversed(range(self.rounds_results_container.count())):
            layout_item = self.rounds_results_container.itemAt(i)
            if layout_item:
                widget = layout_item.widget()
                if widget:
                    widget.setParent(None)
                    widget.deleteLater()

    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def show_error(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("Ошибка")
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #333333;
                color: white;
            }
             QMessageBox QLabel {
                color: white;
            }

            QMessageBox QPushButton{
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #008080, stop: 1 #006666);
                color: white;
                padding: 5px;
            }
            QMessageBox QPushButton:hover{
                background-color: #006666;
                }
            """)

        msg_box.exec_()

    def closeEvent(self, event):
        if self.hand_tracking_thread:
            self.hand_tracking_thread.stop()
        self.countdown_timer.stop()
        self.delay_timer.stop()
        event.accept()

    def get_stylesheet(self):
        return """
                        QWidget {
                            background-color: #222222;
                            color: #FFFFFF;
                        }
                        QLabel {
                            color: #FFFFFF;
                        }
                        QPushButton {
                            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                stop: 0 #008080, stop: 1 #006666);
                            color: white;
                            border: none;
                            padding: 10px 20px;
                            border-radius: 15px;
                            font-size: 18px;
                        }
                        QPushButton:hover {
                            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                stop: 0 #006666, stop: 1 #004d4d);
                        }
                        QPushButton:pressed {
                            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                                    stop: 0  #004d4d, stop: 1 #003333);
                        }
                        QPushButton:disabled {
                            background-color: #666666;
                            color: #999999;
                        }
                        QLineEdit {
                            color: white;
                            background-color: #333333;
                            border: 2px solid #555555;
                            border-radius: 10px;
                            padding: 8px;
                            selection-background-color: #008080;
                        }
                        QLineEdit:focus {
                            border: 2px solid #008080;
                        }
                        """


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
            QMessageBox {
                background-color: #333333;
                color: white;
            }
             QMessageBox QLabel {
                color: white;
            }

            QMessageBox QPushButton{
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #008080, stop: 1 #006666);
                color: white;
                padding: 5px;
            }
            QMessageBox QPushButton:hover{
                background-color: #006666;
                }
            """)
    try:
        game = RockPaperScissorsGame()
        game.showFullScreen()
        sys.exit(app.exec_())
    except Exception as main_error:
        print(f"Unhandled exception in main application: {main_error}")
        import traceback
        traceback.print_exc()
        sys.exit(-1)