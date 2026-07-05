### 🚗 Vehicle Speed Detection 

A real-time Vehicle Speed Detection System built with 

- **YOLOv8n**
- **ByteTrack**
- **OpenCV**
- **Streamlit**

This application detects vehicles in uploaded videos, tracks them across frames, estimates their speed using perspective transformation, and displays the processed video with speed annotations.

---
## Table of Contents

- <a href="#Features">Features</a>
- <a href="#Technologies Used">Technologies Used</a>
- <a href="#Project Structure">Project Structure</a>
- <a href="#Installation">Installation</a>
- <a href="#How to Run">How to Run</a>
- <a href="#How to Use">How to Use</a>
- <a href="#Speed Estimation Method">Speed Estimation Method</a>
- <a href="#Output">Output</a>
- <a href="#Future Improvements">Future Improvements</a>
- <a href="#Author">Author</a>

---
<h2><a class="anchor" id="Features"></a>Features</h2>

- 🚗 Detects multiple vehicle types:
    - Car
    - Motorcycle
    - Bus
    - Truck
- Vehicle tracking using ByteTrack
- Perspective transformation for accurate distance estimation
- Speed estimation in km/h
- Upload videos through a Streamlit web interface
- Download processed video
- Simple and user-friendly interface

---
<h2><a class="anchor" id="Technologies Used"></a>Technologies Used</h2>

- Python	
- YOLOv8n
- ByteTrack	
- OpenCV	
- NumPy	
- Supervision	
- Streamlit

--- 
<h2><a class="anchor" id="Project Structure"></a>Project Structure</h2>

📁Vehicle-Speed-Detection/
│
├── app.py                      # Streamlit application
├── speed_vehical_detector.py   # Detection and speed calculation
├── uploads/                    # Uploaded videos
├── outputs/                    # Processed videos
├── requirements.txt
├── README.md
└── yolov8n.pt                  # YOLO model (download automatically if not present)          

--- 
<h2><a class="anchor" id="Installation"></a>Installation</h2>

- Clone the Repository
git clone https://github.com/yourusername/Vehicle-Speed-Detection.git

cd Vehicle-Speed-Detection

- Create Virtual Environment (Optional)

Windows

python -m venv venv

venv\Scripts\activate

Linux / Mac

python3 -m venv venv

source venv/bin/activate

- Install Dependencies
pip install -r requirements.txty

--- 
<h2><a class="anchor" id="How to Run"></a>How to Run</h2>

- streamlit run app.py

--- 
<h2><a class="anchor" id="How to Use"></a>How to Use</h2>

- Launch the Streamlit application.
- Upload a road traffic video.
- Wait for processing.
- View the processed video.
- Download the output video.

---
<h2><a class="anchor" id="Speed Estimation Method"></a>Speed Estimation Method</h2>

- The system estimates vehicle speed using the following pipeline:

    - Detect vehicles using YOLOv8.
    - Track each vehicle using ByteTrack.
    - Apply Perspective Transformation to convert image coordinates into bird's-eye view.
    - Calculate displacement between frames.
    - Convert pixel movement into real-world distance.
    - Compute speed using:
      - Speed = Distance / Time

The final speed is displayed in km/h.

--- 
<h2><a class="anchor" id="Output"></a>Output</h2>

- The processed video displays:

    - Bounding Boxes
    - Vehicle IDs
    - Vehicle Trails
    - Estimated Speed (km/h)

- Example:

    #12 58 km/h
    #18 42 km/h
    #25 71 km/h

--- 
<h2><a class="anchor" id="Future Improvements"></a>Future Improvements</h2>

- Vehicle counting
- Lane detection
- Traffic density analysis
- Automatic speed limit violation detection
- Number plate recognition (ANPR)
- Export speed reports as CSV
- Live webcam support
- Multiple camera support

----
## Author

Ram Krishna
- Email: ramkrishna000888@gmail.com
- Linkeddin: https://www.linkedin.com/in/ramkrishna000/

