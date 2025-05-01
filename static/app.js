// DOM Elements
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const webcamFeed = document.getElementById('webcam-feed');
const statusText = document.getElementById('status-text');
const loader = document.getElementById('loader');

let isCameraActive = false;

function updateStatus(message, color = '#00d9ff') {
    statusText.textContent = message;
    statusText.style.color = color;
}

function toggleLoader(show) {
    loader.style.display = show ? 'block' : 'none';
}

async function startCamera() {
    try {
        toggleLoader(true);
        startBtn.disabled = true;
        stopBtn.disabled = true;
        updateStatus("Starting camera...");

        const response = await fetch('/start_camera');
        const data = await response.json();

        if (data.status === "Camera started" || data.status.includes("already")) {
            webcamFeed.src = '/video';
            webcamFeed.style.display = 'block';
            isCameraActive = true;
            updateStatus("Camera is running...");
        } else {
            updateStatus("Failed to start camera.", "red");
        }
    } catch (error) {
        console.error('Error starting camera:', error);
        updateStatus("Error starting camera. Try again.", "red");
    } finally {
        toggleLoader(false);
        startBtn.disabled = false;
        stopBtn.disabled = !isCameraActive;
    }
}

async function stopCamera() {
    try {
        toggleLoader(true);
        startBtn.disabled = true;
        stopBtn.disabled = true;
        updateStatus("Stopping camera...");

        const response = await fetch('/stop_camera');
        const data = await response.json();

        if (data.status === "Camera stopped") {
            webcamFeed.src = '';
            webcamFeed.style.display = 'none';
            isCameraActive = false;
            updateStatus("Camera stopped.");
        } else {
            updateStatus("Failed to stop camera.", "red");
        }
    } catch (error) {
        console.error('Error stopping camera:', error);
        updateStatus("Error stopping camera. Try again.", "red");
    } finally {
        toggleLoader(false);
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

// Event Listeners
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

// Initialize UI
document.addEventListener('DOMContentLoaded', () => {
    webcamFeed.style.display = 'none';
    stopBtn.disabled = true;
    updateStatus("Ready to start!");
});
