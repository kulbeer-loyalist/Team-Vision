const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const controlsElement = document.getElementsByClassName('control-panel')[0];
const canvasCtx = canvasElement.getContext('2d');

const fpsControl = new FPS();

const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
  spinner.style.display = 'none';
};

function removeElements(landmarks, elements) {
  for (const element of elements) {
    delete landmarks[element];
  }
}

function removeLandmarks(results) {
  if (results.poseLandmarks) {
    removeElements(
      results.poseLandmarks,
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    );
  }
}

function connect(ctx, connectors) {
  const canvas = ctx.canvas;
  for (const connector of connectors) {
    const from = connector[0];
    const to = connector[1];
    if (from && to) {
      if (
        from.visibility && to.visibility &&
        (from.visibility < 0.1 || to.visibility < 0.1)
      ) {
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
      ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
      ctx.stroke();
    }
  }
}

function onResults(results) {
  document.body.classList.add('loaded');

  removeLandmarks(results);

  fpsControl.tick();

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
    results.image, 0, 0, canvasElement.width, canvasElement.height
  );

  const face_keep_points = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37];
  face_keep_points.sort();
  
  const left_hand_keep_points = Array.from({ length: 21 }, (_, i) => i);
  const pose_keep_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24];
  const right_hand_keep_points = Array.from({ length: 21 }, (_, i) => i);
  
  const face_keep_idx = face_keep_points.map((val) => face_keep_points[val]);
  const left_hand_keep_idx = left_hand_keep_points.map((val) => val + 468);
  const pose_keep_idx = pose_keep_points.map((val) => val + 468 + 21);
  const right_hand_keep_idx = right_hand_keep_points.map((val) => val + 468 + 21 + 33);
  
  const landmarks_to_keep = [
    ...face_keep_idx,
    ...left_hand_keep_idx,
    ...pose_keep_idx,
    ...right_hand_keep_idx,
  ];
  // Filter results.poseLandmarks to keep only the desired landmarks
  results.poseLandmarks = results.poseLandmarks.filter((_, idx) =>
    landmarks_to_keep.includes(idx)
  );

  drawConnectors(
    canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
    { color: '#00FF00' }
  );

  // ... (other draw functions remain unchanged)

  canvasCtx.restore();
}

function startWebcamAndDetection() {

    const holistic = new Holistic({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.1/${file}`;
      }
    });
    holistic.onResults(onResults);
    let webcamEnabled = false;
    
    const camera = new Camera(videoElement, {
      onFrame: async () => {
        await holistic.send({ image: videoElement });
      },
    });
    camera.start();
  
    new ControlPanel(controlsElement, {
      selfieMode: true,
      upperBodyOnly: true,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    }).add([
      new StaticText({ title: 'MediaPipe Holistic' }),
      fpsControl,
      new Toggle({ title: 'Selfie Mode', field: 'selfieMode' }),
      new Toggle({ title: 'Upper-body Only', field: 'upperBodyOnly' }),
      new Toggle({ title: 'Smooth Landmarks', field: 'smoothLandmarks' }),
      new Slider({
        title: 'Min Detection Confidence',
        field: 'minDetectionConfidence',
        range: [0, 1],
        step: 0.01,
      }),
      new Slider({
        title: 'Min Tracking Confidence',
        field: 'minTrackingConfidence',
        range: [0, 1],
        step: 0.01,
      }),
    ]).on(options => {
      videoElement.classList.toggle('selfie', options.selfieMode);
      holistic.setOptions(options);
    });
  }
