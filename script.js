let modelSession;
const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const fpsElement = document.getElementById("fps");
const startButton = document.getElementById("startButton");

// Colors for bounding boxes (matching your Python script)
const bboxColors = [
    [164, 120, 87], [68, 148, 228], [93, 97, 209], [178, 182, 133],
    [88, 159, 106], [96, 202, 231], [159, 124, 168], [169, 162, 241],
    [98, 118, 150], [172, 176, 184]
];

// Load ONNX model
async function loadModel() {
    modelSession = await ort.InferenceSession.create("my_model.onnx");
    console.log("Model loaded!");
}

// Preprocess frame (resize, normalize)
function preprocessFrame(frame) {
    const img = cv2.resize(frame, new cv2.Size(640, 640));
    const tensor = cv2.dnn.blobFromImage(img, 1/255.0);
    return tensor;
}

// Draw bounding boxes
function drawBoxes(detections, frameWidth, frameHeight) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let objectCount = 0;

    detections.forEach(det => {
        const [xmin, ymin, xmax, ymax, conf, classId] = det;
        if (conf > 0.5) {
            const color = bboxColors[classId % 10];
            ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
            ctx.lineWidth = 2;
            ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

            // Draw label
            ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
            ctx.font = "14px Arial";
            ctx.fillText(`${classId}: ${(conf * 100).toFixed(1)}%`, xmin, ymin - 5);
            objectCount++;
        }
    });

    return objectCount;
}

// Main detection loop
async function detectObjects() {
    if (!modelSession) return;

    const startTime = performance.now();
    
    // Capture frame from webcam
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Run inference (simplified)
    const inputTensor = new ort.Tensor("float32", new Float32Array(imageData.data), [1, 3, 640, 640]);
    const { output } = await modelSession.run({ images: inputTensor });
    
    // Process output (mocked - replace with your NMS logic)
    const detections = processYOLOOutput(output); 
    const count = drawBoxes(detections, canvas.width, canvas.height);
    
    // Calculate FPS
    const fps = 1000 / (performance.now() - startTime);
    fpsElement.textContent = fps.toFixed(2);
    
    requestAnimationFrame(detectObjects);
}

// Start webcam
startButton.addEventListener("click", async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            detectObjects();
        };
    } catch (err) {
        console.error("Webcam error:", err);
    }
});

// Initialize
loadModel();
