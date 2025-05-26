from flask import Flask, render_template, Response, request
import cv2
import threading
from ultralytics import YOLO

app = Flask(__name__)

# Global variables
video_active = False
current_frame = None
lock = threading.Lock()
model = YOLO('weights/new8n(25ep).pt')
CONFIDENCE_THRESHOLD = 0.65

def video_processing():
    global video_active, current_frame
    cap = cv2.VideoCapture(1) # change the value to 0 to use laptop's webcam. 1 is for droid cam
    
    while video_active:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO processing
        results = model.predict(frame, verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = box.conf.item()  # Convert tensor to float
                    if conf >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = model.names[int(box.cls[0])]
                        # Format text with converted confidence
                        text = f"{label} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, text, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Update current frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        with lock:
            current_frame = jpeg.tobytes()
    
    cap.release()
    with lock:
        current_frame = None

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/test')
def test():
    return "Server is working!"

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if current_frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                             'Pragma': 'no-cache',
                             'Expires': '0'})  # Add these headers to forcefully ignore socket connections

@app.route('/control', methods=['POST'])
def control():
    global video_active
    action = request.json['action']
    
    if action == 'start' and not video_active:
        video_active = True
        threading.Thread(target=video_processing).start()
    elif action == 'stop' and video_active:
        video_active = False
        
    return {'status': 'success'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)