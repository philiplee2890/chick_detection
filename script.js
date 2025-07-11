// Initialize variables
let model = null;
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const fpsDisplay = document.getElementById('fps');

// Colors for bounding boxes
const bboxColors = [
  [164, 120, 87], [68, 148, 228], [93, 97, 209],
  [178, 182, 133], [88, 159, 106], [96, 202, 231],
  [159, 124, 168], [169, 162, 241], [98, 118, 150],
  [172, 176, 184]
];

// Class labels (replace with your actual classes)
const labels = ['chick', 'sickchick']; // Example labels

async function init() {
  try {
    // 1. Load model
    model = await loadModel();
    
    // 2. Start webcam
    await startWebcam();
    
    // 3. Begin detection loop
    detectObjects();
  } catch (e) {
    console.error("Initialization failed:", e);
    alert("Error initializing: " + e.message);
  }
}

async function loadModel() {
  try {
    // For external models (Google Drive/S3):
    const modelUrl = 'https://drive.google.com/uc?export=download&id=15Q41vuto_IeyDwfnaWtUypLh_qk0j-MB';

    return await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm']
    });
  } catch (e) {
    console.error("Model loading failed:", e);
    throw new Error("Failed to load model. See console for details.");
  }
}

async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'environment'
      },
      audio: false
    });
    
    video.srcObject = stream;
    
    // Wait for video metadata to load
    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        resolve();
      };
    });
  } catch (e) {
    console.error("Webcam error:", e);
    throw new Error("Could not access webcam. Please enable permissions.");
  }
}

function processOutput(output) {
  // Mock implementation - replace with your actual NMS processing
  // This should return an array of detections in format:
  // [x, y, width, height, confidence, classId]
  return [
    [100, 100, 200, 200, 0.9, 0], // Example detection
    [300, 150, 100, 100, 0.85, 2]  // Another example
  ];
}

function drawBoxes(detections) {
  // Clear previous frame
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Draw current video frame
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  // Draw each detection
  detections.forEach(det => {
    const [x, y, w, h, conf, classId] = det;
    
    if (conf > 0.5) { // Confidence threshold
      const color = bboxColors[classId % bboxColors.length];
      
      // Draw bounding box
      ctx.strokeStyle = `rgb(${color.join(',')})`;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      
      // Draw label background
      const label = `${labels[classId]} ${(conf * 100).toFixed(1)}%`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = `rgba(${color.join(',')}, 0.5)`;
      ctx.fillRect(x, y - 20, textWidth + 10, 20);
      
      // Draw label text
      ctx.fillStyle = 'white';
      ctx.font = '12px Arial';
      ctx.fillText(label, x + 5, y - 5);
    }
  });
}

let lastTimestamp = performance.now();
let frameCount = 0;
let fps = 0;

async function detectObjects() {
  try {
    if (!model || !video.videoWidth) return;
    
    // 1. Run inference
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = new ort.Tensor('float32', imageData.data, [1, 3, canvas.height, canvas.width]);
    const { output } = await model.run({ images: inputTensor });
    
    // 2. Process results
    const detections = processOutput(output);
    
    // 3. Draw results
    drawBoxes(detections);
    
    // 4. Calculate FPS
    frameCount++;
    const now = performance.now();
    if (now - lastTimestamp >= 1000) {
      fps = frameCount;
      frameCount = 0;
      lastTimestamp = now;
    }
    fpsDisplay.textContent = `FPS: ${fps}`;
    
    // 5. Repeat
    requestAnimationFrame(detectObjects);
  } catch (e) {
    console.error("Detection error:", e);
  }
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', init);
