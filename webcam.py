from flask import Flask, render_template, Response, request, jsonify
import cv2
import torch
import time
import threading
import function.utils_rotate as utils_rotate
import function.helper as helper

app = Flask(__name__)

# =================== Load YOLO models ===================
print("[INFO] Loading YOLO models...")
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

# =================== Biến toàn cục ===================
current_rtsp = None
video_capture = None
lock = threading.Lock()
plates_list = []
prev_time = 0

def process_stream():
    """Luồng xử lý camera RTSP và nhận diện"""
    global video_capture, plates_list, prev_time

    while True:
        if video_capture is None:
            time.sleep(0.2)
            continue

        success, frame = video_capture.read()
        if not success:
            time.sleep(0.1)
            continue

        # === Nhận diện biển số ===
        plates = yolo_LP_detect(frame, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()

        for plate in list_plates:
            x, y, x2, y2 = map(int, plate[:4])
            crop_img = frame[y:y2, x:x2]
            lp = "unknown"

            for cc in range(0, 2):
                for ct in range(0, 2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        with lock:
                            if lp not in plates_list:
                                plates_list.append(lp)
                        cv2.putText(frame, lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)
                        break
                if lp != "unknown":
                    break

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)

        # === FPS ===
        new_time = time.time()
        fps = 1 / (new_time - prev_time + 1e-5)
        prev_time = new_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # === Cập nhật frame hiển thị ===
        with lock:
            global latest_frame
            latest_frame = frame.copy()

def gen_frames():
    """Truyền frame sang trình duyệt (MJPEG)"""
    global latest_frame
    while True:
        with lock:
            if 'latest_frame' in globals() and latest_frame is not None:
                frame = latest_frame.copy()
            else:
                frame = None
        if frame is None:
            time.sleep(0.1)
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_stream():
    """Nhận URL RTSP từ form"""
    global current_rtsp, video_capture, plates_list

    rtsp_url = request.form.get('rtsp_url')
    if not rtsp_url:
        return jsonify({'status': 'error', 'msg': 'Chưa nhập URL RTSP'}), 400

    with lock:
        plates_list = []
        current_rtsp = rtsp_url
        if video_capture is not None:
            video_capture.release()
        video_capture = cv2.VideoCapture(rtsp_url)

    return jsonify({'status': 'ok', 'msg': 'Đã khởi động luồng camera!'})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plates')
def get_plates():
    """Trả về danh sách biển số nhận diện"""
    with lock:
        return jsonify({'plates': plates_list})

if __name__ == '__main__':
    # Thread xử lý camera song song
    threading.Thread(target=process_stream, daemon=True).start()
    print("[INFO] Server chạy tại: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
