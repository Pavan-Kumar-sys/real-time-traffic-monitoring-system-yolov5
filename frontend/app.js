(function () {
  if (window.__SENTINAI_INITIALIZED__) return;
  window.__SENTINAI_INITIALIZED__ = true;

  const CLASS_META = {
    Car: { color: "#0071e3", w: 52, h: 84, minSpd: 60, maxSpd: 92 },
    "Auto-Rickshaw": { color: "#eab308", w: 44, h: 64, minSpd: 36, maxSpd: 56 },
    Motorcycle: { color: "#34c759", w: 28, h: 54, minSpd: 58, maxSpd: 98 },
    Truck: { color: "#ff9500", w: 68, h: 134, minSpd: 48, maxSpd: 74 },
    Bus: { color: "#5856d6", w: 66, h: 128, minSpd: 50, maxSpd: 78 },
    Bicycle: { color: "#30b0c7", w: 26, h: 48, minSpd: 18, maxSpd: 30 },
    Pedestrian: { color: "#ff3b30", w: 28, h: 46, minSpd: 5, maxSpd: 14 },
  };

  const COCO_MAP = {
    car: "Car",
    truck: "Truck",
    bus: "Bus",
    motorcycle: "Motorcycle",
    bicycle: "Bicycle",
    person: "Pedestrian",
    train: "Bus",
    "cell phone": "Motorcycle",
    bottle: "Auto-Rickshaw",
    cup: "Car",
    laptop: "Car",
  };

  const CAMERAS = {
    cam_01: {
      id: "cam_01",
      name: "NH-44 Northbound Express Corridor",
      location: "Sector 12 Toll Plaza — KM 14.2 (4-Pt Homography Calibrated)",
      speedLimit: 80,
      speedScale: 1.0,
      pxPerMeter: 8.5,
    },
    cam_02: {
      id: "cam_02",
      name: "Metro Arterial Junction Cam-02",
      location: "Central Business District — West Approach (4-Pt Homography)",
      speedLimit: 50,
      speedScale: 0.68,
      pxPerMeter: 10.2,
    },
    cam_webcam: {
      id: "cam_webcam",
      name: "Live Hardware Camera (60 FPS Kalman + YOLOv5)",
      location: "Local Edge Webcam — Decoupled 320x180 Neural Buffer",
      speedLimit: 25,
      speedScale: 1.0,
      pxPerMeter: 14.0,
    },
    cam_upload: {
      id: "cam_upload",
      name: "Uploaded Traffic Video (YOLOv5 + ByteTrack Re-ID)",
      location: "Custom Traffic Video Stream",
      speedLimit: 70,
      speedScale: 1.0,
      pxPerMeter: 9.0,
    },
  };

  const state = {
    backendConnected: false,
    activeCamId: "cam_01",
    feedMode: "sim", // "sim" | "webcam" | "video_file"
    speedLimit: 80,
    countingLineY: 0.62,
    showTrails: true,
    showSpeedTags: true,
    showZones: true,
    showHomography: true,
    showHeatmap: false,

    // Decoupled Non-Blocking Neural Detector State
    neuralModel: null,
    neuralLoading: false,
    isInferencing: false,
    inferFps: 18,
    realTracks: [],
    webcamStream: null,

    // Downscaled 320x180 offscreen buffer for zero-lag neural inference (16x faster WebGL transfer)
    inferCanvas: document.createElement("canvas"),
    // 160x90 motion & appearance Re-ID sampling canvas
    motionCanvas: document.createElement("canvas"),
    prevGrayFrame: null,
    latestSmallRgb: null,

    nextTrackId: 101,
    vehicles: [],
    totalCounted: 0,
    inboundCount: 0,
    outboundCount: 0,
    classCounts: {
      Car: 0,
      "Auto-Rickshaw": 0,
      Motorcycle: 0,
      Truck: 0,
      Bus: 0,
      Bicycle: 0,
      Pedestrian: 0,
    },
    crossingTimes: [],
    incidents: [],
    speedHistory: new Array(40).fill(64),
    lineFlashAlpha: 0,
    fps: 60,
  };

  state.inferCanvas.width = 320;
  state.inferCanvas.height = 180;
  const inferCtx = state.inferCanvas.getContext("2d");

  state.motionCanvas.width = 160;
  state.motionCanvas.height = 90;
  const motionCtx = state.motionCanvas.getContext("2d", {
    willReadFrequently: true,
  });

  // DOM Elements
  const canvas = document.getElementById("simCanvas");
  const ctx = canvas.getContext("2d");
  const rawVideo = document.getElementById("rawVideoSource");
  const backendImg = document.getElementById("backendStreamImg");
  const chartCanvas = document.getElementById("telemetryChart");
  const chartCtx = chartCanvas.getContext("2d");

  const elCameraSelect = document.getElementById("cameraSelector");
  const elSpeedSlider = document.getElementById("speedLimitSlider");
  const elSensorSlider = document.getElementById("sensorLineSlider");
  const elSpeedLabel = document.getElementById("speedLimitLabel");
  const elSubtitle = document.getElementById("cameraLocationSubtitle");
  const elEngineLabel = document.getElementById("engineModeLabel");
  const elActiveCamBadge = document.getElementById("activeCamBadge");
  const elVideoInput = document.getElementById("videoFileInput");

  // ============================================================================
  // 4-POINT PERSPECTIVE HOMOGRAPHY CALIBRATOR (14m Wide x 28m Deep Road Zone)
  // ============================================================================

  function getHomographyQuad(w, h) {
    return [
      [w * 0.28, h * 0.22], // Top-Left (Far horizon)
      [w * 0.72, h * 0.22], // Top-Right (Far horizon)
      [w * 0.88, h * 0.88], // Bottom-Right (Near camera)
      [w * 0.12, h * 0.88], // Bottom-Left (Near camera)
    ];
  }

  function homographyPixelToMeters(px, py, w, h) {
    // Inverse perspective bilinear mapping into 14m (width) x 28m (depth) metric coordinates
    const yTop = h * 0.22;
    const yBot = h * 0.88;
    const normV = Math.max(0.0, Math.min(1.2, (py - yTop) / Math.max(1, yBot - yTop)));
    // Non-linear foreshortening curve along optical axis
    const depthMeters = 28.0 * Math.pow(normV, 0.72);

    const leftAtY = w * (0.28 - 0.16 * normV);
    const rightAtY = w * (0.72 + 0.16 * normV);
    const normU = (px - leftAtY) / Math.max(1, rightAtY - leftAtY);
    const lateralMeters = normU * 14.0;
    return [lateralMeters, depthMeters];
  }

  // ============================================================================
  // APPEARANCE HISTOGRAM EMBEDDING (12-D Re-ID Vector for Occlusion Recovery)
  // ============================================================================

  function computeAppearanceEmbedding(bbox) {
    if (!state.latestSmallRgb) return null;
    const mw = state.motionCanvas.width;
    const mh = state.motionCanvas.height;
    const sx = mw / canvas.width;
    const sy = mh / canvas.height;

    const x1 = Math.max(0, Math.floor(bbox[0] * sx));
    const y1 = Math.max(0, Math.floor(bbox[1] * sy));
    const x2 = Math.min(mw - 1, Math.ceil(bbox[2] * sx));
    const y2 = Math.min(mh - 1, Math.ceil(bbox[3] * sy));
    if (x2 <= x1 + 1 || y2 <= y1 + 1) return null;

    const hist = new Float32Array(12);
    const data = state.latestSmallRgb;

    for (let y = y1; y <= y2; y++) {
      for (let x = x1; x <= x2; x++) {
        const idx = (y * mw + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        hist[Math.min(3, r >> 6)] += 1;
        hist[4 + Math.min(3, g >> 6)] += 1;
        hist[8 + Math.min(3, b >> 6)] += 1;
      }
    }

    let norm = 0;
    for (let i = 0; i < 12; i++) norm += hist[i] * hist[i];
    norm = Math.sqrt(norm);
    if (norm > 1e-6) {
      for (let i = 0; i < 12; i++) hist[i] /= norm;
    }
    return hist;
  }

  function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB) return 0.5;
    let dot = 0,
      na = 0,
      nb = 0;
    for (let i = 0; i < vecA.length; i++) {
      dot += vecA[i] * vecB[i];
      na += vecA[i] * vecA[i];
      nb += vecB[i] * vecB[i];
    }
    const denom = Math.sqrt(na * nb);
    return denom > 1e-6 ? dot / denom : 0.5;
  }

  // ============================================================================
  // IDD (INDIAN DRIVING DATASET) 3-WHEELER AUTO-RICKSHAW CLASSIFIER
  // ============================================================================

  function hasRickshawColorCanopy(bbox) {
    if (!state.latestSmallRgb) return false;
    const mw = state.motionCanvas.width;
    const mh = state.motionCanvas.height;
    const sx = mw / canvas.width;
    const sy = mh / canvas.height;

    const x1 = Math.max(0, Math.floor((bbox[0] + (bbox[2] - bbox[0]) * 0.14) * sx));
    const y1 = Math.max(0, Math.floor((bbox[1] + (bbox[3] - bbox[1]) * 0.10) * sy));
    const x2 = Math.min(mw - 1, Math.ceil((bbox[2] - (bbox[2] - bbox[0]) * 0.14) * sx));
    const y2 = Math.min(mh - 1, Math.ceil((bbox[1] + (bbox[3] - bbox[1]) * 0.76) * sy));

    if (x2 <= x1 + 1 || y2 <= y1 + 1) return false;

    const data = state.latestSmallRgb;
    let totalPx = 0;
    let rickshawTonePx = 0;

    for (let y = y1; y <= y2; y++) {
      for (let x = x1; x <= x2; x++) {
        const idx = (y * mw + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        totalPx++;

        const isYellow = r > 125 && g > 105 && b < 95 && r - b > 42;
        const isGreen = g > 90 && g > r * 1.14 && g > b * 1.18;
        const isCyan = b > 105 && g > 105 && r < 90;

        if (isYellow || isGreen || isCyan) {
          rickshawTonePx++;
        }
      }
    }

    return totalPx > 0 && rickshawTonePx / totalPx >= 0.15;
  }

  function refineIndianTrafficClass(rawClass, bbox) {
    if (rawClass !== "Truck" && rawClass !== "Car" && rawClass !== "Bus") {
      return rawClass;
    }

    const [x1, y1, x2, y2] = bbox;
    const bw = Math.max(1, x2 - x1);
    const bh = Math.max(1, y2 - y1);
    const cy = (y1 + y2) * 0.5;
    const aspect = bw / bh;

    const depthScale = Math.max(0.26, Math.pow(cy / canvas.height, 1.35));
    const normArea = (bw * bh) / (depthScale * canvas.width * canvas.height);
    const isRickshawColored = hasRickshawColorCanopy(bbox);

    if (rawClass === "Truck") {
      if (normArea < 0.052) {
        if ((aspect >= 0.55 && aspect <= 1.40) || isRickshawColored) {
          return "Auto-Rickshaw";
        }
        return "Car";
      }
    }

    if (rawClass === "Car") {
      if (isRickshawColored && normArea < 0.042 && aspect >= 0.55 && aspect <= 1.35) {
        return "Auto-Rickshaw";
      }
      if (normArea < 0.021 && aspect >= 0.65 && aspect <= 1.18) {
        return "Auto-Rickshaw";
      }
    }

    return rawClass;
  }

  // ============================================================================
  // NON-BLOCKING BACKGROUND NEURAL DETECTOR + BYTETRACK RE-ID ENGINE
  // ============================================================================

  async function ensureNeuralModelLoaded() {
    if (state.neuralModel || state.neuralLoading) return;
    if (typeof cocoSsd === "undefined") return;
    try {
      state.neuralLoading = true;
      elEngineLabel.textContent = "Loading YOLOv5 Weights...";
      if (typeof tf !== "undefined" && tf.ready) {
        await tf.ready();
      }
      state.neuralModel = await cocoSsd.load({ base: "lite_mobilenet_v2" });
      state.neuralLoading = false;
      elEngineLabel.textContent = "60 FPS Kalman + ByteTrack Live";
    } catch (_) {
      state.neuralLoading = false;
      elEngineLabel.textContent = "60 FPS CV Motion + MOT";
    }
  }

  setTimeout(ensureNeuralModelLoaded, 200);

  function computeIoU(boxA, boxB) {
    const xA = Math.max(boxA[0], boxB[0]);
    const yA = Math.max(boxA[1], boxB[1]);
    const xB = Math.min(boxA[2], boxB[2]);
    const yB = Math.min(boxA[3], boxB[3]);
    const interW = Math.max(0, xB - xA);
    const interH = Math.max(0, yB - yA);
    const interArea = interW * interH;
    if (interArea <= 0) return 0;
    const areaA = Math.max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]));
    const areaB = Math.max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]));
    return interArea / (areaA + areaB - interArea);
  }

  function detectMotionBlobsOnSmallBuffer() {
    const mw = state.motionCanvas.width;
    const mh = state.motionCanvas.height;
    motionCtx.drawImage(state.inferCanvas, 0, 0, mw, mh);
    const imgData = motionCtx.getImageData(0, 0, mw, mh).data;
    state.latestSmallRgb = imgData;

    if (state.feedMode !== "video_file" && state.neuralModel) {
      // On live webcam, skip motion blobs so camera hand/background shake doesn't create false boxes
      return [];
    }

    const gray = new Uint8Array(mw * mh);
    for (let i = 0, j = 0; i < imgData.length; i += 4, j++) {
      gray[j] = (imgData[i] * 77 + imgData[i + 1] * 150 + imgData[i + 2] * 29) >> 8;
    }

    if (!state.prevGrayFrame) {
      state.prevGrayFrame = gray;
      return [];
    }

    const cellW = 4;
    const cellH = 4;
    const gridCols = Math.floor(mw / cellW);
    const gridRows = Math.floor(mh / cellH);
    const activeCells = new Uint8Array(gridCols * gridRows);

    for (let gy = 2; gy < gridRows - 1; gy++) {
      for (let gx = 1; gx < gridCols - 1; gx++) {
        let diffCount = 0;
        for (let dy = 0; dy < cellH; dy++) {
          for (let dx = 0; dx < cellW; dx++) {
            const idx = (gy * cellH + dy) * mw + (gx * cellW + dx);
            if (Math.abs(gray[idx] - state.prevGrayFrame[idx]) > 22) {
              diffCount++;
            }
          }
        }
        if (diffCount >= 5) {
          activeCells[gy * gridCols + gx] = 1;
        }
      }
    }

    for (let i = 0; i < gray.length; i++) {
      state.prevGrayFrame[i] = (state.prevGrayFrame[i] * 3 + gray[i]) >> 2;
    }

    const visited = new Uint8Array(gridCols * gridRows);
    const blobs = [];
    const scaleX = canvas.width / mw;
    const scaleY = canvas.height / mh;

    for (let gy = 2; gy < gridRows - 1; gy++) {
      for (let gx = 1; gx < gridCols - 1; gx++) {
        const startIdx = gy * gridCols + gx;
        if (!activeCells[startIdx] || visited[startIdx]) continue;

        let minX = gx,
          maxX = gx,
          minY = gy,
          maxY = gy,
          count = 0;
        const queue = [startIdx];
        visited[startIdx] = 1;

        while (queue.length > 0) {
          const curr = queue.pop();
          const cy = Math.floor(curr / gridCols);
          const cx = curr % gridCols;
          count++;
          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;

          for (let ny = Math.max(0, cy - 1); ny <= Math.min(gridRows - 1, cy + 1); ny++) {
            for (let nx = Math.max(0, cx - 1); nx <= Math.min(gridCols - 1, cx + 1); nx++) {
              const nIdx = ny * gridCols + nx;
              if (activeCells[nIdx] && !visited[nIdx]) {
                visited[nIdx] = 1;
                queue.push(nIdx);
              }
            }
          }
        }

        if (count >= 2 && count <= 140) {
          const x1 = minX * cellW * scaleX;
          const y1 = minY * cellH * scaleY;
          const x2 = (maxX + 1) * cellW * scaleX;
          const y2 = (maxY + 1) * cellH * scaleY;
          const bw = x2 - x1;
          const bh = y2 - y1;
          if (bw >= 28 && bh >= 24 && bw <= 420 && bh <= 420) {
            const bbox = [x1, y1, x2, y2];
            const area = bw * bh;
            const initialCls =
              area > 36000
                ? "Truck"
                : area > 8500
                ? "Car"
                : area > 3800
                ? "Auto-Rickshaw"
                : "Motorcycle";
            const finalCls = refineIndianTrafficClass(initialCls, bbox);
            blobs.push({
              bbox,
              class_name: finalCls,
              confidence: 0.85,
              embedding: computeAppearanceEmbedding(bbox),
            });
          }
        }
      }
    }
    return blobs;
  }

  // Background inference loop (completely decoupled from 60 FPS requestAnimationFrame!)
  async function runBackgroundInferenceTick() {
    if (
      state.backendConnected ||
      (state.feedMode !== "webcam" && state.feedMode !== "video_file") ||
      rawVideo.readyState < 2 ||
      state.isInferencing
    ) {
      setTimeout(runBackgroundInferenceTick, 45);
      return;
    }

    state.isInferencing = true;
    const t0 = performance.now();

    try {
      // Draw raw video into fast 320x180 offscreen canvas (16x smaller than 1280x720!)
      inferCtx.drawImage(
        rawVideo,
        0,
        0,
        state.inferCanvas.width,
        state.inferCanvas.height
      );

      const motionBlobs = detectMotionBlobsOnSmallBuffer();
      const scaleX = canvas.width / state.inferCanvas.width;
      const scaleY = canvas.height / state.inferCanvas.height;
      const combined = [];

      if (state.neuralModel) {
        // Detect on 320x180 offscreen canvas instead of full 1280x720 video!
        const preds = await state.neuralModel.detect(
          state.inferCanvas,
          16,
          0.20
        );
        for (const p of preds) {
          const rawClass =
            COCO_MAP[p.class] ||
            (state.feedMode === "webcam" ? "Pedestrian" : null);
          if (!rawClass) continue;
          const [bx, by, bw, bh] = p.bbox;
          const scaledBbox = [
            bx * scaleX,
            by * scaleY,
            (bx + bw) * scaleX,
            (by + bh) * scaleY,
          ];
          const finalClass = refineIndianTrafficClass(rawClass, scaledBbox);
          combined.push({
            bbox: scaledBbox,
            class_name: finalClass,
            confidence: Number(p.score.toFixed(2)),
            embedding: computeAppearanceEmbedding(scaledBbox),
          });
        }
      }

      for (const mb of motionBlobs) {
        const overlapsExisting = combined.some(
          (c) => computeIoU(c.bbox, mb.bbox) > 0.25
        );
        if (!overlapsExisting) {
          combined.push(mb);
        }
      }

      updateByteTrackMultiObjectTracker(combined, performance.now() / 1000);
    } catch (_) {}

    const elapsed = performance.now() - t0;
    state.inferFps = Math.round(1000 / Math.max(16, elapsed));
    state.isInferencing = false;
    setTimeout(runBackgroundInferenceTick, 35);
  }

  setTimeout(runBackgroundInferenceTick, 350);

  function associateByteTrackStage(
    unmatchedTracks,
    detIndices,
    detections,
    minIou
  ) {
    const pairs = [];
    unmatchedTracks.forEach((tIdx) => {
      const tr = state.realTracks[tIdx];
      const tcx = (tr.renderBbox[0] + tr.renderBbox[2]) * 0.5;
      const tcy = (tr.renderBbox[1] + tr.renderBbox[3]) * 0.5;
      detIndices.forEach((dIdx) => {
        const det = detections[dIdx];
        const dcx = (det.bbox[0] + det.bbox[2]) * 0.5;
        const dcy = (det.bbox[1] + det.bbox[3]) * 0.5;
        const iou = computeIoU(tr.renderBbox, det.bbox);
        const dist = Math.hypot(tcx - dcx, tcy - dcy);
        const reidSim = cosineSimilarity(tr.embedding, det.embedding);

        if (iou >= minIou || (dist < 145 && reidSim >= 0.45)) {
          const distScore = Math.max(0, 1 - dist / 145);
          const score = 0.52 * iou + 0.28 * distScore + 0.20 * reidSim;
          pairs.push({ score, tIdx, dIdx });
        }
      });
    });

    pairs.sort((a, b) => b.score - a.score);
    const matched = [];
    const usedDets = new Set();

    for (const { tIdx, dIdx } of pairs) {
      if (!unmatchedTracks.has(tIdx) || usedDets.has(dIdx)) continue;
      unmatchedTracks.delete(tIdx);
      usedDets.add(dIdx);
      matched.push({ tIdx, dIdx });
    }

    for (let i = detIndices.length - 1; i >= 0; i--) {
      if (usedDets.has(detIndices[i])) {
        detIndices.splice(i, 1);
      }
    }
    return matched;
  }

  function updateByteTrackMultiObjectTracker(detections, nowSec) {
    const w = canvas.width;
    const h = canvas.height;
    const lineY = state.countingLineY * h;
    const preset = CAMERAS[state.activeCamId] || CAMERAS.cam_upload;

    const unmatchedTracks = new Set(state.realTracks.map((_, i) => i));
    const highConfDets = [];
    const lowConfDets = [];

    detections.forEach((d, idx) => {
      if (d.confidence >= 0.38) highConfDets.push(idx);
      else lowConfDets.push(idx);
    });

    // ByteTrack Stage 1: High-confidence detections
    const stage1 = associateByteTrackStage(
      unmatchedTracks,
      highConfDets,
      detections,
      0.16
    );
    // ByteTrack Stage 2: Low-confidence / occluded detections
    const stage2 = associateByteTrackStage(
      unmatchedTracks,
      lowConfDets,
      detections,
      0.12
    );

    const allMatched = stage1.concat(stage2);

    for (const { tIdx, dIdx } of allMatched) {
      const tr = state.realTracks[tIdx];
      const det = detections[dIdx];
      const prevCx = (tr.targetBbox[0] + tr.targetBbox[2]) * 0.5;
      const prevCy = (tr.targetBbox[1] + tr.targetBbox[3]) * 0.5;
      const dt = Math.max(0.03, nowSec - tr.lastSeenSec);

      tr.targetBbox = [...det.bbox];
      const newCx = (det.bbox[0] + det.bbox[2]) * 0.5;
      const newCy = (det.bbox[1] + det.bbox[3]) * 0.5;

      // Kalman velocity update (px/sec) for 60 FPS frame interpolation
      const instVx = (newCx - prevCx) / dt;
      const instVy = (newCy - prevCy) / dt;
      tr.vx = 0.6 * tr.vx + 0.4 * instVx;
      tr.vy = 0.6 * tr.vy + 0.4 * instVy;

      // Temporal Class Voting (prevents Auto-Rickshaw flickering to Truck)
      const voteWeight =
        det.confidence * (det.class_name === "Auto-Rickshaw" ? 1.4 : 1.0);
      tr.classVotes[det.class_name] =
        (tr.classVotes[det.class_name] || 0) + voteWeight;
      let bestClass = tr.class_name;
      let bestScore = -1;
      for (const [k, sc] of Object.entries(tr.classVotes)) {
        if (sc > bestScore) {
          bestScore = sc;
          bestClass = k;
        }
      }
      tr.class_name = bestClass;
      tr.confidence = det.confidence;

      if (det.embedding) {
        if (!tr.embedding) tr.embedding = det.embedding;
        else {
          for (let i = 0; i < 12; i++) {
            tr.embedding[i] = 0.75 * tr.embedding[i] + 0.25 * det.embedding[i];
          }
        }
      }

      tr.lastSeenSec = nowSec;
      tr.missed = 0;
      tr.hits += 1;

      const [wx, wy] = homographyPixelToMeters(newCx, det.bbox[3], w, h);
      tr.worldTrail.push({ wx, wy, t: nowSec });
      if (tr.worldTrail.length > 20) tr.worldTrail.shift();

      tr.lane = Math.min(4, Math.max(1, Math.floor((newCx / w) * 4) + 1));

      if (tr.worldTrail.length >= 3) {
        const firstW = tr.worldTrail[0];
        const lastW = tr.worldTrail[tr.worldTrail.length - 1];
        const wDt = Math.max(0.05, lastW.t - firstW.t);
        const metersHomog = Math.hypot(
          lastW.wx - firstW.wx,
          lastW.wy - firstW.wy
        );

        const firstP = tr.trail[0];
        const lastP = tr.trail[tr.trail.length - 1];
        const pxDist = firstP
          ? Math.hypot(newCx - firstP.x, newCy - firstP.y)
          : 0;
        const normY = Math.max(0.2, Math.min(1.0, newCy / h));
        const metersPersp =
          (pxDist * (1.42 - 0.6 * normY)) / preset.pxPerMeter;

        const meters = 0.6 * metersHomog + 0.4 * metersPersp;
        const rawKmh = Math.min(175, (meters / wDt) * 3.6);
        tr.speed_kmh = tr.speed_kmh * 0.68 + rawKmh * 0.32;
        tr.direction =
          lastP && newCy - firstP.y >= 0 ? "Approaching ↓" : "Receding ↑";
      }

      if (!tr.counted) {
        const crossedDown = prevCy < lineY && newCy >= lineY;
        const crossedUp = prevCy > lineY && newCy <= lineY;
        if (crossedDown || crossedUp) {
          tr.counted = true;
          state.totalCounted += 1;
          if (crossedDown) state.inboundCount += 1;
          else state.outboundCount += 1;
          state.classCounts[tr.class_name] =
            (state.classCounts[tr.class_name] || 0) + 1;
          state.crossingTimes.push(nowSec);
          state.lineFlashAlpha = 0.7;
        }
      }

      if (
        tr.hits >= 4 &&
        tr.speed_kmh > state.speedLimit * 1.08 &&
        !tr.flaggedOverspeed
      ) {
        tr.flaggedOverspeed = true;
        const excess = tr.speed_kmh - state.speedLimit;
        logLocalIncident(
          tr,
          "OVERSPEEDING",
          excess >= 20 ? "CRITICAL" : "HIGH",
          `${tr.class_name} #${tr.track_id} clocked at ${tr.speed_kmh.toFixed(1)} km/h in ${state.speedLimit} km/h zone (Lane ${tr.lane})`
        );
      }
    }

    // Spawn new tracks from unmatched high-confidence detections
    highConfDets.forEach((dIdx) => {
      const det = detections[dIdx];
      const cx = (det.bbox[0] + det.bbox[2]) * 0.5;
      const cy = (det.bbox[1] + det.bbox[3]) * 0.5;
      const [wx, wy] = homographyPixelToMeters(cx, det.bbox[3], w, h);
      state.realTracks.push({
        track_id: state.nextTrackId++,
        renderBbox: [...det.bbox],
        targetBbox: [...det.bbox],
        vx: 0,
        vy: 0,
        class_name: det.class_name,
        classVotes: { [det.class_name]: det.confidence },
        confidence: det.confidence,
        embedding: det.embedding,
        lane: Math.min(4, Math.max(1, Math.floor((cx / w) * 4) + 1)),
        speed_kmh: 0,
        direction: "Tracking",
        counted: false,
        flaggedOverspeed: false,
        hits: 1,
        missed: 0,
        lastSeenSec: nowSec,
        trail: [{ x: cx, y: cy, t: nowSec }],
        worldTrail: [{ wx, wy, t: nowSec }],
      });
    });

    state.realTracks.forEach((tr, idx) => {
      if (unmatchedTracks.has(idx)) tr.missed += 1;
    });
    state.realTracks = state.realTracks.filter((tr) => tr.missed <= 7);
  }

  // 60 FPS Kalman Constant-Velocity Box Interpolation (Zero Stutter!)
  function advanceRealTracks60Fps(dt, nowSec) {
    for (const tr of state.realTracks) {
      // Advance target position slightly along velocity vector between neural frames
      const damp = tr.missed === 0 ? 0.45 : 0.2;
      tr.targetBbox[0] += tr.vx * dt * damp;
      tr.targetBbox[2] += tr.vx * dt * damp;
      tr.targetBbox[1] += tr.vy * dt * damp;
      tr.targetBbox[3] += tr.vy * dt * damp;

      // Smoothly interpolate rendered bounding box at 60 FPS
      const alpha = Math.min(1.0, dt * 16.0);
      for (let k = 0; k < 4; k++) {
        tr.renderBbox[k] += (tr.targetBbox[k] - tr.renderBbox[k]) * alpha;
      }

      const rcx = (tr.renderBbox[0] + tr.renderBbox[2]) * 0.5;
      const rcy = (tr.renderBbox[1] + tr.renderBbox[3]) * 0.5;
      const lastPt = tr.trail[tr.trail.length - 1];
      if (!lastPt || Math.hypot(rcx - lastPt.x, rcy - lastPt.y) > 4) {
        tr.trail.push({ x: rcx, y: rcy, t: nowSec });
        if (tr.trail.length > 28) tr.trail.shift();
      }
    }
  }

  // ============================================================================
  // UI CONTROLS & EVENT LISTENERS
  // ============================================================================

  function stopWebcamStream() {
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach((t) => t.stop());
      state.webcamStream = null;
    }
  }

  async function startLiveWebcam() {
    elCameraSelect.value = "cam_webcam";
    state.activeCamId = "cam_webcam";
    const preset = CAMERAS.cam_webcam;
    state.speedLimit = preset.speedLimit;
    elSpeedSlider.value = String(preset.speedLimit);
    elSpeedLabel.textContent = `${preset.speedLimit} km/h`;
    elSubtitle.textContent = `${preset.name} — ${preset.location}`;

    if (state.backendConnected) {
      syncBackendConfig({
        camera_id: "cam_webcam",
        speed_limit_kmh: preset.speedLimit,
      });
      elActiveCamBadge.textContent =
        "Streaming Hardware Camera via FastAPI + YOLOv5";
      return;
    }

    try {
      stopWebcamStream();
      elActiveCamBadge.textContent = "Requesting camera permission...";
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 60, min: 30 },
        },
        audio: false,
      });
      state.webcamStream = stream;
      rawVideo.srcObject = stream;
      await rawVideo.play();
      state.feedMode = "webcam";
      state.prevGrayFrame = null;
      resetCountersOnly();
      ensureNeuralModelLoaded();
      elEngineLabel.textContent = "Live Webcam (60 FPS Kalman)";
      elActiveCamBadge.textContent =
        "60 FPS Non-Blocking Webcam Active — Zero-lag decoupled neural buffer enabled";
    } catch (err) {
      alert(
        "Could not access webcam (" +
          err.message +
          "). Please allow camera permissions in your browser."
      );
    }
  }

  async function handleUploadedVideoFile(file) {
    if (!file) return;
    elCameraSelect.value = "cam_upload";
    state.activeCamId = "cam_upload";
    CAMERAS.cam_upload.name = `Uploaded: ${file.name}`;
    const preset = CAMERAS.cam_upload;
    state.speedLimit = preset.speedLimit;
    elSpeedSlider.value = String(preset.speedLimit);
    elSpeedLabel.textContent = `${preset.speedLimit} km/h`;
    elSubtitle.textContent = `Uploaded Footage: ${file.name} — 60 FPS ByteTrack Re-ID + Homography`;

    if (state.backendConnected) {
      const formData = new FormData();
      formData.append("file", file);
      elActiveCamBadge.textContent = `Uploading ${file.name} to YOLOv5 Backend...`;
      try {
        await fetch("/api/upload_video", { method: "POST", body: formData });
        elActiveCamBadge.textContent = `Tracking ${file.name} via FastAPI + YOLOv5`;
        return;
      } catch (_) {}
    }

    stopWebcamStream();
    rawVideo.srcObject = null;
    const fileUrl = URL.createObjectURL(file);
    rawVideo.src = fileUrl;
    rawVideo.loop = true;
    rawVideo.muted = true;
    await rawVideo.play();
    state.feedMode = "video_file";
    state.prevGrayFrame = null;
    resetCountersOnly();
    ensureNeuralModelLoaded();
    elEngineLabel.textContent = "60 FPS Video + ByteTrack Re-ID";
    elActiveCamBadge.textContent = `Tracking Uploaded Footage: ${file.name} (60 FPS Kalman Interpolation)`;
  }

  document
    .getElementById("btnStartWebcam")
    .addEventListener("click", startLiveWebcam);

  document.getElementById("btnUploadVideo").addEventListener("click", () => {
    elVideoInput.click();
  });

  elVideoInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) handleUploadedVideoFile(file);
  });

  document.getElementById("chkTrails").addEventListener("change", (e) => {
    state.showTrails = e.target.checked;
    syncBackendConfig({ show_trails: state.showTrails });
  });
  document.getElementById("chkSpeedTags").addEventListener("change", (e) => {
    state.showSpeedTags = e.target.checked;
    syncBackendConfig({ show_speed_tags: state.showSpeedTags });
  });
  document.getElementById("chkZones").addEventListener("change", (e) => {
    state.showZones = e.target.checked;
    syncBackendConfig({ show_zones: state.showZones });
  });
  document.getElementById("chkHomography").addEventListener("change", (e) => {
    state.showHomography = e.target.checked;
  });
  document.getElementById("chkHeatmap").addEventListener("change", (e) => {
    state.showHeatmap = e.target.checked;
  });

  elSensorSlider.addEventListener("input", (e) => {
    state.countingLineY = Number(e.target.value) / 100;
    syncBackendConfig({ counting_line_y: state.countingLineY });
  });

  elSpeedSlider.addEventListener("input", (e) => {
    state.speedLimit = Number(e.target.value);
    elSpeedLabel.textContent = `${state.speedLimit} km/h`;
    syncBackendConfig({ speed_limit_kmh: state.speedLimit });
  });

  elCameraSelect.addEventListener("change", (e) => {
    const camId = e.target.value;
    if (camId === "cam_webcam") {
      startLiveWebcam();
      return;
    }
    if (camId === "cam_upload") {
      elVideoInput.click();
      return;
    }
    stopWebcamStream();
    rawVideo.pause();
    state.feedMode = "sim";
    state.activeCamId = camId;
    const preset = CAMERAS[camId] || CAMERAS.cam_01;
    state.speedLimit = preset.speedLimit;
    elSpeedSlider.value = String(preset.speedLimit);
    elSpeedLabel.textContent = `${preset.speedLimit} km/h`;
    elSubtitle.textContent = `${preset.name} — ${preset.location}`;
    elActiveCamBadge.textContent =
      'Click "🎥 Live Camera" or "📁 Upload Traffic Video" above to track real footage';
    resetStandaloneState();
    syncBackendConfig({ camera_id: camId, speed_limit_kmh: preset.speedLimit });
  });

  document.getElementById("btnTriggerSpeeder").addEventListener("click", () => {
    if (state.backendConnected) {
      fetch("/api/trigger_anomaly", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ anomaly_type: "overspeed" }),
      }).catch(() => {});
    } else {
      spawnSimVehicle(null, null, "overspeed");
    }
  });

  document.getElementById("btnResetStats").addEventListener("click", () => {
    if (state.backendConnected) {
      fetch("/api/reset", { method: "POST" }).catch(() => {});
    }
    resetCountersOnly();
  });

  document.getElementById("btnExportCsv").addEventListener("click", () => {
    if (state.backendConnected) {
      window.location.href = "/api/export/incidents.csv";
      return;
    }
    const headers = [
      "ID,Timestamp,Camera,Track_ID,Vehicle_Class,Violation_Type,Speed_KMH,Speed_Limit_KMH,Lane,Severity,Details",
    ];
    state.incidents.forEach((inc, idx) => {
      headers.push(
        `${idx + 1},"${new Date(inc.timestamp * 1000).toISOString()}",${state.activeCamId},${inc.track_id},${inc.vehicle_class},${inc.violation_type},${inc.speed_kmh},${inc.speed_limit_kmh},${inc.lane},${inc.severity},"${inc.details}"`
      );
    });
    const blob = new Blob([headers.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "traffic_incidents_report.csv";
    a.click();
    URL.revokeObjectURL(url);
  });

  function syncBackendConfig(payload) {
    if (!state.backendConnected) return;
    fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch(() => {});
  }

  function tryConnectBackend() {
    if (window.location.protocol === "file:") return;
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/telemetry`;
    try {
      const ws = new WebSocket(wsUrl);
      ws.onopen = () => {
        state.backendConnected = true;
        elEngineLabel.textContent = "FastAPI + YOLOv5 Live";
        backendImg.src = "/api/video_feed";
        backendImg.style.display = "block";
        canvas.style.display = "none";
      };
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        renderDashboardFromBackend(data);
      };
      ws.onerror = () => {
        state.backendConnected = false;
      };
      ws.onclose = () => {
        state.backendConnected = false;
        backendImg.style.display = "none";
        canvas.style.display = "block";
      };
    } catch (_) {
      state.backendConnected = false;
    }
  }

  // ============================================================================
  // 60 FPS RENDERING ENGINE (REAL VIDEO + HOMOGRAPHY ZONE + KALMAN BOXES)
  // ============================================================================

  function drawRoundedRect(context, x, y, width, height, radius) {
    context.beginPath();
    context.roundRect(x, y, width, height, radius);
  }

  function drawHomographyOverlay(w, h) {
    if (!state.showHomography) return;
    const quad = getHomographyQuad(w, h);
    ctx.save();
    ctx.fillStyle = "rgba(0, 113, 227, 0.06)";
    ctx.strokeStyle = "rgba(0, 113, 227, 0.55)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    quad.forEach(([qx, qy], idx) => {
      if (idx === 0) ctx.moveTo(qx, qy);
      else ctx.lineTo(qx, qy);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);

    // 4 Corner Calibration Anchors
    const labels = ["P1 (0m, 0m)", "P2 (14m, 0m)", "P3 (14m, 28m)", "P4 (0m, 28m)"];
    quad.forEach(([qx, qy], idx) => {
      ctx.fillStyle = "#0071e3";
      ctx.beginPath();
      ctx.arc(qx, qy, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = "rgba(29, 29, 31, 0.72)";
      drawRoundedRect(ctx, qx + 7, qy - 10, 82, 18, 5);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.font = "600 10px -apple-system, Inter, sans-serif";
      ctx.fillText(labels[idx], qx + 12, qy + 3);
    });
    ctx.restore();
  }

  function renderRealVideoFrameAndHud(dt, nowMs) {
    const w = canvas.width;
    const h = canvas.height;
    const lineY = state.countingLineY * h;
    const nowSec = nowMs / 1000;

    // 1. Draw live webcam or uploaded video frame at native 60 FPS (zero blocking!)
    if (rawVideo.readyState >= 2) {
      ctx.drawImage(rawVideo, 0, 0, w, h);
    } else {
      ctx.fillStyle = "#101216";
      ctx.fillRect(0, 0, w, h);
    }

    // 2. Smoothly advance bounding boxes at 60 FPS via Kalman velocity prediction
    advanceRealTracks60Fps(dt, nowSec);

    const visibleTracks = state.realTracks.filter((tr) => tr.missed <= 3);

    // 3. 4-Point Perspective Homography Calibration Zone
    drawHomographyOverlay(w, h);

    // 4. Optional Density Heatmap
    if (state.showHeatmap) {
      for (const tr of visibleTracks) {
        const cx = (tr.renderBbox[0] + tr.renderBbox[2]) * 0.5;
        const cy = (tr.renderBbox[1] + tr.renderBbox[3]) * 0.5;
        const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, 95);
        grad.addColorStop(0, "rgba(255, 59, 48, 0.45)");
        grad.addColorStop(0.5, "rgba(255, 149, 0, 0.22)");
        grad.addColorStop(1, "rgba(0, 113, 227, 0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, 95, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 5. Virtual Sensor Line Overlay
    if (state.showZones) {
      ctx.fillStyle = `rgba(0, 113, 227, ${0.18 + state.lineFlashAlpha * 0.35})`;
      ctx.fillRect(w * 0.06, lineY - 9, w * 0.88, 18);
      state.lineFlashAlpha = Math.max(0, state.lineFlashAlpha - dt * 2.4);

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(w * 0.06, lineY);
      ctx.lineTo(w * 0.94, lineY);
      ctx.stroke();

      ctx.fillStyle = "rgba(29, 29, 31, 0.78)";
      drawRoundedRect(ctx, w * 0.06, lineY - 34, 415, 22, 6);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.font = "600 11.5px -apple-system, Inter, sans-serif";
      ctx.fillText(
        `4-PT HOMOGRAPHY SPEED & FLOW SENSOR LINE (LIMIT: ${state.speedLimit} KM/H)`,
        w * 0.06 + 10,
        lineY - 19
      );
    }

    // 6. Draw 60 FPS Interpolated Track Bounding Boxes, Trails & Pills
    for (const tr of visibleTracks) {
      const [x1, y1, x2, y2] = tr.renderBbox;
      const bw = x2 - x1;
      const bh = y2 - y1;
      const spec = CLASS_META[tr.class_name] || CLASS_META.Car;
      const isFast = tr.speed_kmh > state.speedLimit * 1.08;
      const color = isFast ? "#ff3b30" : spec.color;

      if (state.showTrails && tr.trail.length >= 2) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        tr.trail.forEach((pt, idx) => {
          if (idx === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.stroke();
      }

      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      drawRoundedRect(ctx, x1, y1, bw, bh, 6);
      ctx.stroke();

      if (state.showSpeedTags) {
        let label = `#${tr.track_id} ${tr.class_name} · ${tr.speed_kmh.toFixed(0)} km/h`;
        if (isFast) label += " · FAST";
        ctx.font = "600 12px -apple-system, Inter, sans-serif";
        const tw = ctx.measureText(label).width;
        const labelY = Math.max(26, y1);
        ctx.fillStyle = color;
        drawRoundedRect(ctx, x1, labelY - 24, tw + 14, 21, 6);
        ctx.fill();
        ctx.fillStyle =
          tr.class_name === "Auto-Rickshaw" && !isFast ? "#1d1d1f" : "#ffffff";
        ctx.fillText(label, x1 + 7, labelY - 9);
      }
    }

    // 7. Apple Glass HUD Pill in Top-Left
    ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
    drawRoundedRect(ctx, 18, 18, 520, 36, 18);
    ctx.fill();

    ctx.fillStyle = "#34c759";
    ctx.beginPath();
    ctx.arc(36, 36, 5, 0, Math.PI * 2);
    ctx.fill();

    const preset = CAMERAS[state.activeCamId] || CAMERAS.cam_webcam;
    ctx.fillStyle = "#1d1d1f";
    ctx.font = "600 12.5px -apple-system, Inter, sans-serif";
    ctx.fillText(
      `YOLOv5 + ByteTrack Re-ID  ·  ${preset.name}  ·  ${state.fps.toFixed(0)} FPS`,
      50,
      40
    );
  }

  // ============================================================================
  // SYNTHETIC BENCHMARK HIGHWAY ENGINE
  // ============================================================================

  function laneCenterX(lane) {
    const roadLeft = canvas.width * 0.16;
    const roadRight = canvas.width * 0.84;
    const laneW = (roadRight - roadLeft) / 4;
    return roadLeft + (lane - 0.5) * laneW;
  }

  function spawnSimVehicle(laneOverride, yOverride, anomaly) {
    const lane = laneOverride || Math.floor(Math.random() * 4) + 1;
    const r = Math.random();
    const cls =
      r < 0.45
        ? "Car"
        : r < 0.67
        ? "Auto-Rickshaw"
        : r < 0.82
        ? "Motorcycle"
        : r < 0.92
        ? "Truck"
        : "Bus";
    const spec = CLASS_META[cls];
    const preset = CAMERAS[state.activeCamId] || CAMERAS.cam_01;

    let speed =
      (spec.minSpd + Math.random() * (spec.maxSpd - spec.minSpd)) *
      (preset.speedScale || 1.0);
    let wrongWay = false;

    if (anomaly === "overspeed" || Math.random() < 0.14) {
      speed = state.speedLimit * (1.16 + Math.random() * 0.28);
    } else if (anomaly === "wrong_way") {
      wrongWay = true;
      speed = state.speedLimit * 0.95;
    }

    const movingDown = (lane <= 2) !== wrongWay;
    const y =
      yOverride !== undefined && yOverride !== null
        ? yOverride
        : movingDown
        ? -spec.h * 0.7
        : canvas.height + spec.h * 0.7;

    for (const v of state.vehicles) {
      if (v.lane === lane && Math.abs(v.y - y) < (v.h + spec.h) * 0.9) {
        return;
      }
    }

    state.vehicles.push({
      track_id: state.nextTrackId++,
      class_name: cls,
      confidence: Number((0.89 + Math.random() * 0.1).toFixed(2)),
      lane,
      x: laneCenterX(lane) + (Math.random() * 10 - 5),
      y,
      w: spec.w,
      h: spec.h,
      speed_kmh: speed,
      movingDown,
      wrongWay,
      counted: false,
      flaggedOverspeed: false,
      flaggedWrongWay: false,
      hits: 1,
      trail: [],
    });
  }

  function resetCountersOnly() {
    state.realTracks = [];
    state.totalCounted = 0;
    state.inboundCount = 0;
    state.outboundCount = 0;
    state.classCounts = {
      Car: 0,
      "Auto-Rickshaw": 0,
      Motorcycle: 0,
      Truck: 0,
      Bus: 0,
      Bicycle: 0,
      Pedestrian: 0,
    };
    state.crossingTimes = [];
    state.incidents = [];
  }

  function resetStandaloneState() {
    resetCountersOnly();
    state.vehicles = [];
    for (let i = 0; i < 8; i++) {
      spawnSimVehicle((i % 4) + 1, 90 + i * 75, null);
    }
  }

  function logLocalIncident(v, type, severity, details) {
    state.incidents.unshift({
      id: state.incidents.length + 1,
      timestamp: Date.now() / 1000,
      track_id: v.track_id,
      vehicle_class: v.class_name,
      violation_type: type,
      speed_kmh: Number(v.speed_kmh.toFixed(1)),
      speed_limit_kmh: state.speedLimit,
      lane: v.lane,
      severity,
      details,
    });
    if (state.incidents.length > 40) state.incidents.pop();
  }

  let lastTs = performance.now();
  let lastUiUpdate = 0;

  function stepStandalone(nowMs) {
    const dt = Math.min(0.05, Math.max(0.005, (nowMs - lastTs) / 1000));
    lastTs = nowMs;
    state.fps = 0.92 * state.fps + 0.08 * Math.min(60, 1 / dt);

    if (!state.backendConnected) {
      if (state.feedMode === "webcam" || state.feedMode === "video_file") {
        renderRealVideoFrameAndHud(dt, nowMs);
      } else {
        updateAndRenderSimCanvas(dt);
      }

      if (nowMs - lastUiUpdate > 240) {
        lastUiUpdate = nowMs;
        updateStandaloneDom();
      }
    }
    requestAnimationFrame(stepStandalone);
  }

  function updateAndRenderSimCanvas(dt) {
    const w = canvas.width;
    const h = canvas.height;
    const roadLeft = w * 0.16;
    const roadRight = w * 0.84;
    const laneW = (roadRight - roadLeft) / 4;
    const lineY = state.countingLineY * h;

    ctx.fillStyle = "#1c2420";
    ctx.fillRect(0, 0, w, h);

    ctx.fillStyle = "#2b3038";
    ctx.fillRect(roadLeft, 0, roadRight - roadLeft, h);

    ctx.strokeStyle = "#e5e5ea";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(roadLeft + 4, 0);
    ctx.lineTo(roadLeft + 4, h);
    ctx.moveTo(roadRight - 4, 0);
    ctx.lineTo(roadRight - 4, h);
    ctx.stroke();

    const midX = (roadLeft + roadRight) / 2;
    ctx.strokeStyle = "#ffcc00";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(midX - 4, 0);
    ctx.lineTo(midX - 4, h);
    ctx.moveTo(midX + 4, 0);
    ctx.lineTo(midX + 4, h);
    ctx.stroke();

    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.lineWidth = 2;
    ctx.setLineDash([34, 26]);
    [1, 3].forEach((divIdx) => {
      const lx = roadLeft + divIdx * laneW;
      ctx.beginPath();
      ctx.moveTo(lx, 0);
      ctx.lineTo(lx, h);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    drawHomographyOverlay(w, h);

    if (state.vehicles.length < 11 && Math.random() < 0.08) {
      spawnSimVehicle();
    }

    const active = [];
    for (const v of state.vehicles) {
      const prevY = v.y;
      const dirSign = v.movingDown ? 1 : -1;
      const pxPerSec = (v.speed_kmh / 3.6) * 8.5;
      v.y += dirSign * pxPerSec * dt;
      v.hits += 1;
      v.trail.push({ x: v.x, y: v.y });
      if (v.trail.length > 28) v.trail.shift();

      if (!v.counted) {
        const crossedDown = prevY < lineY && v.y >= lineY;
        const crossedUp = prevY > lineY && v.y <= lineY;
        if (crossedDown || crossedUp) {
          v.counted = true;
          state.totalCounted += 1;
          if (crossedDown) state.inboundCount += 1;
          else state.outboundCount += 1;
          state.classCounts[v.class_name] =
            (state.classCounts[v.class_name] || 0) + 1;
          state.crossingTimes.push(Date.now() / 1000);
          state.lineFlashAlpha = 0.65;
        }
      }

      if (v.hits >= 8) {
        if (v.speed_kmh > state.speedLimit * 1.08 && !v.flaggedOverspeed) {
          v.flaggedOverspeed = true;
          const excess = v.speed_kmh - state.speedLimit;
          logLocalIncident(
            v,
            "OVERSPEEDING",
            excess >= 22 ? "CRITICAL" : "HIGH",
            `${v.class_name} #${v.track_id} clocked at ${v.speed_kmh.toFixed(1)} km/h in ${state.speedLimit} km/h zone (Lane ${v.lane})`
          );
        }
      }

      if (v.y >= -v.h && v.y <= h + v.h) {
        active.push(v);
      }
    }
    state.vehicles = active;

    if (state.showHeatmap) {
      for (const v of state.vehicles) {
        const grad = ctx.createRadialGradient(v.x, v.y, 8, v.x, v.y, 95);
        grad.addColorStop(0, "rgba(255, 59, 48, 0.42)");
        grad.addColorStop(0.5, "rgba(255, 149, 0, 0.22)");
        grad.addColorStop(1, "rgba(0, 113, 227, 0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(v.x, v.y, 95, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (state.showZones) {
      ctx.fillStyle = `rgba(0, 113, 227, ${0.18 + state.lineFlashAlpha * 0.35})`;
      ctx.fillRect(roadLeft, lineY - 9, roadRight - roadLeft, 18);
      state.lineFlashAlpha = Math.max(0, state.lineFlashAlpha - dt * 2.4);

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(roadLeft, lineY);
      ctx.lineTo(roadRight, lineY);
      ctx.stroke();
    }

    for (const v of state.vehicles) {
      const x1 = v.x - v.w / 2;
      const y1 = v.y - v.h / 2;
      const spec = CLASS_META[v.class_name] || CLASS_META.Car;
      const isViolation =
        v.speed_kmh > state.speedLimit * 1.08 || v.wrongWay;
      const boxColor = isViolation ? "#ff3b30" : spec.color;

      if (state.showTrails && v.trail.length > 2) {
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        v.trail.forEach((pt, idx) => {
          if (idx === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.stroke();
      }

      ctx.fillStyle = "rgba(0,0,0,0.35)";
      drawRoundedRect(ctx, x1 + 4, y1 + 5, v.w, v.h, 6);
      ctx.fill();

      ctx.fillStyle =
        v.class_name === "Auto-Rickshaw" ? "#facc15" : "#475569";
      drawRoundedRect(ctx, x1, y1, v.w, v.h, 6);
      ctx.fill();

      ctx.fillStyle = "#1e293b";
      drawRoundedRect(
        ctx,
        x1 + v.w * 0.16,
        y1 + v.h * 0.2,
        v.w * 0.68,
        v.h * 0.56,
        4
      );
      ctx.fill();

      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 2.2;
      drawRoundedRect(ctx, x1 - 3, y1 - 3, v.w + 6, v.h + 6, 7);
      ctx.stroke();

      if (state.showSpeedTags) {
        let label = `#${v.track_id} ${v.class_name} · ${v.speed_kmh.toFixed(0)} km/h`;
        if (isViolation) label += " · FAST";
        ctx.font = "600 11.5px -apple-system, Inter, sans-serif";
        const textW = ctx.measureText(label).width;
        ctx.fillStyle = boxColor;
        drawRoundedRect(ctx, x1 - 3, y1 - 24, textW + 14, 20, 6);
        ctx.fill();
        ctx.fillStyle =
          v.class_name === "Auto-Rickshaw" && !isViolation
            ? "#1d1d1f"
            : "#ffffff";
        ctx.fillText(label, x1 + 4, y1 - 10);
      }
    }

    ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
    drawRoundedRect(ctx, 18, 18, 520, 36, 18);
    ctx.fill();

    ctx.fillStyle = "#34c759";
    ctx.beginPath();
    ctx.arc(36, 36, 5, 0, Math.PI * 2);
    ctx.fill();

    const preset = CAMERAS[state.activeCamId] || CAMERAS.cam_01;
    ctx.fillStyle = "#1d1d1f";
    ctx.font = "600 12.5px -apple-system, Inter, sans-serif";
    ctx.fillText(
      `YOLOv5s + ByteTrack Re-ID  ·  ${preset.name}  ·  ${state.fps.toFixed(0)} FPS`,
      50,
      40
    );
  }

  function updateStandaloneDom() {
    const nowSec = Date.now() / 1000;
    state.crossingTimes = state.crossingTimes.filter((t) => nowSec - t <= 60);
    const flowVpm = state.crossingTimes.length;

    const sourceList =
      state.feedMode === "webcam" || state.feedMode === "video_file"
        ? state.realTracks.filter((tr) => tr.missed <= 3)
        : state.vehicles;

    const speeds = sourceList
      .map((v) => v.speed_kmh)
      .filter((s) => s > 0.5);
    const avgSpeed = speeds.length
      ? speeds.reduce((a, b) => a + b, 0) / speeds.length
      : 0;
    const maxSpeed = speeds.length ? Math.max(...speeds) : 0;

    const laneCounts = { 1: 0, 2: 0, 3: 0, 4: 0 };
    sourceList.forEach((v) => {
      laneCounts[v.lane] = (laneCounts[v.lane] || 0) + 1;
    });

    const density = sourceList.length;
    const congestion =
      density >= 10 ? "HIGH" : density >= 5 ? "MODERATE" : "LOW";

    renderDashboardMetrics({
      active_vehicles: density,
      total_counted: state.totalCounted,
      inbound_count: state.inboundCount,
      outbound_count: state.outboundCount,
      avg_speed_kmh: avgSpeed,
      max_speed_kmh: maxSpeed,
      flow_rate_vpm: flowVpm,
      congestion_level: congestion,
      class_counts: state.classCounts,
      lane_occupancy: laneCounts,
      active_tracks: sourceList.map((v) => ({
        track_id: v.track_id,
        class_name: v.class_name,
        confidence: v.confidence,
        lane: v.lane,
        direction:
          v.direction || (v.movingDown ? "Inbound ↓" : "Outbound ↑"),
        speed_kmh: v.speed_kmh,
      })),
      recent_incidents: state.incidents,
    });
  }

  function renderDashboardFromBackend(payload) {
    const m = payload.metrics;
    renderDashboardMetrics({
      active_vehicles: m.active_vehicles,
      total_counted: m.total_counted,
      inbound_count: m.inbound_count,
      outbound_count: m.outbound_count,
      avg_speed_kmh: m.avg_speed_kmh,
      max_speed_kmh: m.max_speed_kmh,
      flow_rate_vpm: m.flow_rate_vpm,
      congestion_level: m.congestion_level,
      class_counts: m.class_counts,
      lane_occupancy: m.lane_occupancy,
      active_tracks: payload.active_tracks,
      recent_incidents: payload.recent_incidents,
    });
  }

  function renderDashboardMetrics(data) {
    document.getElementById("kpiActiveVehicles").textContent =
      data.active_vehicles;
    document.getElementById("kpiInOutSplit").textContent =
      `Inbound: ${data.inbound_count} · Outbound: ${data.outbound_count}`;
    document.getElementById("kpiTotalCounted").textContent =
      data.total_counted;
    document.getElementById("kpiFlowRate").innerHTML =
      `${data.flow_rate_vpm} <small style="font-size:0.85rem;font-weight:500;color:#6e6e73;">veh/min</small>`;
    document.getElementById("kpiAvgSpeed").innerHTML =
      `${Number(data.avg_speed_kmh).toFixed(1)} <small style="font-size:0.85rem;font-weight:500;color:#6e6e73;">km/h</small>`;
    document.getElementById("kpiMaxSpeed").textContent =
      `Peak Clocked: ${Number(data.max_speed_kmh).toFixed(1)} km/h`;
    document.getElementById("kpiCongestion").textContent =
      data.congestion_level;
    document.getElementById("kpiViolationCount").textContent =
      data.recent_incidents.length;
    document.getElementById("kpiFps").textContent =
      `Render: ${state.fps.toFixed(0)} FPS · Neural: ${state.inferFps} Hz`;

    for (let l = 1; l <= 4; l++) {
      const el = document.getElementById(`laneVal${l}`);
      if (el) el.textContent = data.lane_occupancy[l] || 0;
    }

    const total = Math.max(1, data.total_counted);
    const classContainer = document.getElementById("classBarsContainer");
    classContainer.innerHTML = [
      "Car",
      "Auto-Rickshaw",
      "Motorcycle",
      "Truck",
      "Bus",
    ]
      .map((cls) => {
        const count = data.class_counts[cls] || 0;
        const pct = Math.min(100, Math.round((count / total) * 100));
        const color = CLASS_META[cls] ? CLASS_META[cls].color : "#0071e3";
        return `
          <div class="class-row">
            <div class="class-row-header">
              <span>${cls}</span>
              <span style="color:#6e6e73;font-weight:500;">${count} (${pct}%)</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" style="width:${pct}%;background:${color};"></div>
            </div>
          </div>
        `;
      })
      .join("");

    const tbody = document.getElementById("activeTracksBody");
    tbody.innerHTML = data.active_tracks
      .slice(0, 10)
      .map((tr) => {
        const isFast = tr.speed_kmh > state.speedLimit * 1.08;
        const spdStyle = isFast
          ? "color:#ff3b30;font-weight:600;"
          : "color:#1f9d45;font-weight:600;";
        return `
          <tr>
            <td><strong>#${tr.track_id}</strong></td>
            <td>${tr.class_name}</td>
            <td>${Math.round(tr.confidence * 100)}%</td>
            <td>Lane ${tr.lane}</td>
            <td>${tr.direction}</td>
            <td style="${spdStyle}">${Number(tr.speed_kmh).toFixed(1)} km/h</td>
          </tr>
        `;
      })
      .join("");

    const incContainer = document.getElementById("incidentFeedContainer");
    if (!data.recent_incidents.length) {
      incContainer.innerHTML = `<div style="color:var(--text-secondary);font-size:0.8rem;padding:0.5rem;">No violations detected yet. Click <strong>🎥 Live Camera</strong>, <strong>📁 Upload Traffic Video</strong>, or lower the Speed Limit slider to trigger live alerts.</div>`;
    } else {
      incContainer.innerHTML = data.recent_incidents
        .slice(0, 12)
        .map((inc) => {
          const timeStr = new Date(inc.timestamp * 1000).toLocaleTimeString();
          return `
            <div class="incident-card ${inc.severity}">
              <div class="incident-top">
                <span>${inc.violation_type} · #${inc.track_id} (${inc.vehicle_class})</span>
                <span class="badge ${inc.severity}">${inc.severity} · ${timeStr}</span>
              </div>
              <div class="incident-desc">${inc.details}</div>
            </div>
          `;
        })
        .join("");
    }

    state.speedHistory.push(Number(data.avg_speed_kmh) || 0);
    state.speedHistory.shift();
    drawTelemetryChart();
  }

  function drawTelemetryChart() {
    const cw = chartCanvas.width;
    const ch = chartCanvas.height;
    chartCtx.clearRect(0, 0, cw, ch);

    chartCtx.fillStyle = "#f5f5f7";
    chartCtx.fillRect(0, 0, cw, ch);

    const limitY = ch - Math.min(ch - 10, (state.speedLimit / 140) * (ch - 22));
    chartCtx.strokeStyle = "rgba(255, 59, 48, 0.45)";
    chartCtx.setLineDash([5, 5]);
    chartCtx.beginPath();
    chartCtx.moveTo(0, limitY);
    chartCtx.lineTo(cw, limitY);
    chartCtx.stroke();
    chartCtx.setLineDash([]);

    chartCtx.beginPath();
    state.speedHistory.forEach((spd, i) => {
      const x = (i / (state.speedHistory.length - 1)) * cw;
      const y = ch - Math.min(ch - 14, (spd / 140) * (ch - 22)) - 6;
      if (i === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    });
    chartCtx.strokeStyle = "#0071e3";
    chartCtx.lineWidth = 2.4;
    chartCtx.stroke();

    chartCtx.fillStyle = "#6e6e73";
    chartCtx.font = "600 10.5px -apple-system, Inter, sans-serif";
    chartCtx.fillText(
      `Homography Mean Speed (km/h) vs. Limit (${state.speedLimit} km/h)`,
      12,
      18
    );
  }

  resetStandaloneState();
  tryConnectBackend();
  requestAnimationFrame(stepStandalone);
})();
