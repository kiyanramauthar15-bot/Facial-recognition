/* =====================================================
   FACEVISION
   AI-POWERED REAL-TIME FACE DETECTION
   JavaScript Application
===================================================== */

// =====================================================
// 1. CONFIGURATION
// =====================================================

const MEDIAPIPE_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

const WASM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";

const MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";


// =====================================================
// 2. HTML ELEMENTS
// =====================================================

const video = document.getElementById("video");

const canvas = document.getElementById("overlay");

const ctx = canvas.getContext("2d");

const startButton = document.getElementById("startButton");

const stopButton = document.getElementById("stopButton");

const faceCount = document.getElementById("faceCount");

const analyticsFaces = document.getElementById("analyticsFaces");

const cameraStatus = document.getElementById("cameraStatus");

const cameraDescription =
    document.getElementById("cameraDescription");

const modelStatus = document.getElementById("modelStatus");

const cameraMessage = document.getElementById("cameraMessage");

const liveBadge = document.getElementById("liveBadge");

const cameraViewport =
    document.getElementById("cameraViewport");

const cameraPlaceholder =
    document.getElementById("cameraPlaceholder");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceBar =
    document.getElementById("confidenceBar");

const processingStatus =
    document.getElementById("processingStatus");

const resolution =
    document.getElementById("resolution");

const cameraResolution =
    document.getElementById("cameraResolution");


// =====================================================
// 3. APPLICATION VARIABLES
// =====================================================

let faceDetector = null;

let cameraStream = null;

let animationFrame = null;

let isRunning = false;

let lastVideoTime = -1;


// =====================================================
// 4. ERROR HANDLING
// =====================================================

function showError(message) {

    console.error("FaceVision:", message);

    cameraMessage.textContent = message;

}


// =====================================================
// 5. INITIALIZE MEDIAPIPE
// =====================================================

async function initializeDetector() {

    try {

        modelStatus.textContent = "Loading";

        cameraMessage.textContent =
            "Connecting to MediaPipe AI detection engine...";

        startButton.disabled = true;

        // Load MediaPipe dynamically.

        const visionModule = await import(MEDIAPIPE_URL);

        const FaceDetector = visionModule.FaceDetector;

        const FilesetResolver = visionModule.FilesetResolver;

        if (!FaceDetector || !FilesetResolver) {

            throw new Error(
                "MediaPipe Face Detector components were not found."
            );

        }

        cameraMessage.textContent =
            "Loading computer vision processing engine...";

        // Load WebAssembly files.

        const vision = await FilesetResolver.forVisionTasks(
            WASM_URL
        );

        cameraMessage.textContent =
            "Downloading face detection model...";

        // Create face detection model.

        faceDetector = await FaceDetector.createFromOptions(
            vision,
            {

                baseOptions: {

                    modelAssetPath: MODEL_URL,

                    delegate: "CPU"

                },

                runningMode: "VIDEO",

                minDetectionConfidence: 0.5,

                minSuppressionThreshold: 0.3

            }
        );

        // Model successfully initialized.

        modelStatus.textContent = "Ready";

        startButton.disabled = false;

        cameraMessage.textContent =
            "Face detection system ready. Click Start Camera to begin.";

        console.log("FaceVision initialized successfully.");

    } catch (error) {

        console.error("MediaPipe initialization failed:", error);

        modelStatus.textContent = "Error";

        startButton.disabled = true;

        cameraMessage.textContent =
            "Model loading failed: " +
            (error.message || "Unknown error") +
            ". Check your internet connection and refresh the page.";

    }

}


// =====================================================
// 6. START WEBCAM
// =====================================================

async function startCamera() {

    if (!faceDetector) {

        showError("Face detection model is not ready yet.");

        return;

    }

    if (isRunning) return;

    try {

        cameraMessage.textContent =
            "Requesting webcam permission...";

        // Request webcam access.

        cameraStream = await navigator.mediaDevices.getUserMedia({

            video: {

                facingMode: "user",

                width: {
                    ideal: 1280
                },

                height: {
                    ideal: 720
                }

            },

            audio: false

        });

        video.srcObject = cameraStream;

        await video.play();

        // Set canvas dimensions.

        canvas.width = video.videoWidth;

        canvas.height = video.videoHeight;

        // Activate application.

        isRunning = true;

        lastVideoTime = -1;

        cameraViewport.classList.add("active");

        cameraPlaceholder.style.display = "none";

        startButton.disabled = true;

        stopButton.disabled = false;

        cameraStatus.textContent = "Active";

        cameraDescription.textContent =
            "Webcam connected";

        liveBadge.classList.add("live");

        liveBadge.innerHTML =
            "<span></span> LIVE";

        resolution.textContent =
            `${video.videoWidth} × ${video.videoHeight}`;

        cameraResolution.textContent =
            `${video.videoWidth} × ${video.videoHeight}`;

        processingStatus.textContent = "Running";

        cameraMessage.textContent =
            "Camera active. Detecting faces in real time.";

        console.log("Webcam successfully started.");

        // Start detection.

        detectFaces();

    } catch (error) {

        console.error("Camera initialization failed:", error);

        let message = "Unable to access the webcam.";

        if (error.name === "NotAllowedError") {

            message =
                "Camera permission denied. Allow camera access in your browser settings.";

        } else if (error.name === "NotFoundError") {

            message =
                "No webcam was found on this device.";

        } else if (error.name === "NotReadableError") {

            message =
                "The webcam is already being used by another application.";

        } else if (error.name === "OverconstrainedError") {

            message =
                "The requested camera settings are not supported.";

        }

        showError(message);

        stopCamera();

    }

}


// =====================================================
// 7. REAL-TIME FACE DETECTION
// =====================================================

function detectFaces() {

    if (!isRunning) return;

    try {

        if (
            video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
            video.currentTime !== lastVideoTime
        ) {

            lastVideoTime = video.currentTime;

            // Process the current webcam frame.

            const results = faceDetector.detectForVideo(
                video,
                performance.now()
            );

            const detections = results.detections || [];

            // Clear previous detection rectangles.

            ctx.clearRect(
                0,
                0,
                canvas.width,
                canvas.height
            );

            let highestConfidence = 0;

            // Process every detected face.

            detections.forEach((detection) => {

                const box = detection.boundingBox;

                if (!box) return;

                const x = box.originX;

                const y = box.originY;

                const width = box.width;

                const height = box.height;

                const confidence =
                    detection.categories?.[0]?.score ?? 0;

                highestConfidence = Math.max(
                    highestConfidence,
                    confidence
                );

                // Draw green detection rectangle.

                ctx.strokeStyle = "#42d6ae";

                ctx.lineWidth = Math.max(
                    2,
                    canvas.width / 400
                );

                ctx.strokeRect(
                    x,
                    y,
                    width,
                    height
                );

                // Create face confidence label.

                const label =
                    `FACE ${(confidence * 100).toFixed(0)}%`;

                ctx.font = "bold 13px Arial";

                const labelWidth =
                    ctx.measureText(label).width + 16;

                const labelY = Math.max(0, y - 28);

                ctx.fillStyle = "#42d6ae";

                ctx.fillRect(
                    x,
                    labelY,
                    labelWidth,
                    25
                );

                ctx.fillStyle = "#071b20";

                ctx.fillText(
                    label,
                    x + 8,
                    labelY + 17
                );

            });


            // Update face statistics.

            faceCount.textContent = detections.length;

            analyticsFaces.textContent = detections.length;


            // Update confidence percentage.

            if (detections.length > 0) {

                const percentage = Math.round(
                    highestConfidence * 100
                );

                confidenceValue.textContent =
                    `${percentage}%`;

                confidenceBar.style.width =
                    `${percentage}%`;

            } else {

                confidenceValue.textContent = "--";

                confidenceBar.style.width = "0%";

            }

        }

    } catch (error) {

        console.error("Face detection error:", error);

        cameraMessage.textContent =
            "An error occurred while processing the webcam feed.";

    }


    // Continue processing webcam frames.

    if (isRunning) {

        animationFrame =
            requestAnimationFrame(detectFaces);

    }

}


// =====================================================
// 8. STOP WEBCAM
// =====================================================

function stopCamera() {

    isRunning = false;

    // Cancel animation.

    if (animationFrame !== null) {

        cancelAnimationFrame(animationFrame);

        animationFrame = null;

    }

    // Release webcam tracks.

    if (cameraStream) {

        cameraStream.getTracks().forEach((track) => {

            track.stop();

        });

        cameraStream = null;

    }

    // Disconnect video.

    video.pause();

    video.srcObject = null;

    // Clear detection canvas.

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    // Reset dashboard.

    cameraViewport.classList.remove("active");

    cameraPlaceholder.style.display = "block";

    cameraStatus.textContent = "Offline";

    cameraDescription.textContent =
        "Waiting to start";

    faceCount.textContent = "0";

    analyticsFaces.textContent = "0";

    confidenceValue.textContent = "--";

    confidenceBar.style.width = "0%";

    processingStatus.textContent = "Waiting";

    resolution.textContent = "-- × --";

    cameraResolution.textContent = "--";

    liveBadge.classList.remove("live");

    liveBadge.innerHTML =
        "<span></span> STANDBY";

    startButton.disabled = !faceDetector;

    stopButton.disabled = true;

    if (faceDetector) {

        cameraMessage.textContent =
            "Camera stopped. Click Start Camera to begin again.";

    }

    lastVideoTime = -1;

    console.log("Webcam stopped.");

}


// =====================================================
// 9. BUTTON EVENT LISTENERS
// =====================================================

startButton.addEventListener(
    "click",
    startCamera
);

stopButton.addEventListener(
    "click",
    stopCamera
);


// =====================================================
// 10. CAMERA SUPPORT CHECK
// =====================================================

if (!navigator.mediaDevices?.getUserMedia) {

    modelStatus.textContent = "Check browser";

    showError(
        "Camera access requires HTTPS or localhost. Open this website through Live Server or GitHub Pages."
    );

}


// =====================================================
// 11. CLEAN UP WHEN LEAVING PAGE
// =====================================================

window.addEventListener("pagehide", () => {

    stopCamera();

});


// =====================================================
// 12. START APPLICATION
// =====================================================

initializeDetector();

console.log("FaceVision application started.");