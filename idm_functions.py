# Function to detect people using YOLO
def detect_persons(frame):
    results = yolo_model(frame)  # Run YOLO
    persons = []

    for result in results:
        for box, class_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):  # Extract confidence score
            if int(class_id) == 0:  # Class 0 in COCO is "person"
                x1, y1, x2, y2 = map(int, box[:4])
                persons.append({
                    "bbox": (x1, y1, x2, y2),  # Bounding box
                    "confidence": conf.item()   # Confidence score
                })

    return persons

# Helper function to save a frame
def save_intruder_frame(frame, prefix=""):
    # Save files to local disk
    filename = f"{INTRUDER_FOLDER}/Intruder_{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")
    
    # Optional: Call FIREBASE cloud upload function here
    # handle_intruder_detection(filename, confidence)

# First Burst Function - 5 photos immediately, 300ms apart
def first_burst(frame):
    global state, burst_in_progress
    
    if state != 'stop' or burst_in_progress:
        return  # Already capturing or burst in progress
    
    state = 'start'
    burst_in_progress = True  # Set the flag to indicate burst is in progress
    print("Intruder detected - Starting first burst capture")

    # Background thread for burst capture
    def capture_burst():
        global burst_in_progress
        
        for i in range(5):
            save_intruder_frame(frame, f"{i+1}_burst")
            time.sleep(0.3)  # This sleep only affects the burst thread, not the main loop

        burst_in_progress = False  # Reset flag after burst capture is complete

    # Start a thread for the burst capture
    threading.Thread(target=capture_burst, daemon=True).start()

# Follow-up Function - 1 photo every 2 seconds
def follow_up(frame):
    global state, last_intruder_time, follow_update
    
    if state != 'start':
        return  # No active capture
    
    if time.time() - last_intruder_time >= 2:  # 2 seconds interval
        save_intruder_frame(frame, "followup")
        last_intruder_time = time.time()
    
# Exit Function - stop after 5 seconds of no intruder detected
def exit_capture():
    print("... INSIDE EXIT...")
    global state, follow_update
    
    print("No intruder for 5 seconds - Stopping capture")
    state = 'stop'
    follow_update = False

