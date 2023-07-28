const imageUpload = document.getElementById('imageUpload');
const videoUpload = document.getElementById('videoUpload');
const previewSection = document.getElementById('previewSection');

imageUpload.addEventListener('change', handleImageUpload);
videoUpload.addEventListener('change', handleVideoUpload);

function handleImageUpload(event) {
    previewSection.innerHTML = ''; // Clear previous previews

    for (let i = 0; i < event.target.files.length; i++) {
        const file = event.target.files[i];
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.classList.add('preview-image');
        previewSection.appendChild(img);
    }
}

function handleVideoUpload(event) {
    previewSection.innerHTML = ''; // Clear previous previews

    for (let i = 0; i < event.target.files.length; i++) {
        const file = event.target.files[i];
        const video = document.createElement('video');
        video.src = URL.createObjectURL(file);
        video.classList.add('preview-video');
        video.autoplay = true;
        video.loop = true;
        video.muted = true;
        previewSection.appendChild(video);
    }
}

// Webcam Enable
const webcamButton = document.getElementById('webcamButton');
const webcamPreview = document.getElementById('webcamPreview');

webcamButton.addEventListener('click', enableWebcam);

function enableWebcam() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                webcamPreview.srcObject = stream;
            })
            .catch(function (error) {
                console.error('Error accessing webcam: ', error);
            });
    } else {
        console.error('Webcam access not supported');
    }
}