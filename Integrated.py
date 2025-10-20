import cv2, numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque

# ------------------ Load Models ------------------
seg_model = YOLO("road_seg_model.pt")        # Road segmentation
det_model = YOLO("Street_light_detection_model.pt")              # Streetlight detection

video_path = "C:\\Users\\acer\\Downloads\\101.mp4"
cap = cv2.VideoCapture(video_path)

W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("integrated_output.mp4", fourcc, fps, (W, H))

# DeepSORT tracker
tracker = DeepSort(max_age=15, n_init=3, max_cosine_distance=0.4, nn_budget=100)

# For plotting intensity variation
intensity_history = deque(maxlen=5000)  # keep long history, scrolling view

# ------------------ Helper Functions ------------------
def merge_masks(result, H, W):
    if result.masks is None:
        return np.zeros((H,W), dtype=np.uint8)
    merged = np.zeros((H,W), dtype=np.uint8)
    for m in result.masks.data:
        m = m.cpu().numpy().astype(np.float32)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        merged[m > 0.5] = 1
    return merged

# ------------------ Main Loop ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()

    # -------- Road Segmentation --------
    seg_res = seg_model.predict(source=frame, imgsz=512, conf=0.25, verbose=False)[0]
    mask = merge_masks(seg_res, H, W)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # -------- ROI Box (inside road, above bottom) --------
    mean_intensity = 0.0
    ys, xs = np.where(mask == 1)
    if ys.size > 0 and xs.size > 0:
        x_left, x_right = xs.min(), xs.max()
        y_top = ys.min()         # top border of road
        y_bottom = ys.max()      # bottom border of road

        roi_height = 60  # adjustable
        # Place ROI not at bottom, not at center, but above bottom
        y1 = y_bottom - int((y_bottom - y_top) * 0.33)  
        y2 = min(y1 + roi_height, y_bottom)

        x1, x2 = x_left, x_right
        roi_mask = mask[y1:y2, x1:x2]
        roi = frame[y1:y2, x1:x2]

        if roi.size and roi_mask.sum() > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_intensity = float(gray[roi_mask > 0].mean())

        # Draw ROI on frame
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(overlay, f"I:{mean_intensity:.1f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        intensity_history.append(mean_intensity)

    # -------- Streetlight Detection + Tracking --------
    det_res = det_model(frame, verbose=False)[0]
    track_inputs = []
    if det_res.boxes is not None:
        boxes = det_res.boxes.xyxy.cpu().numpy()
        confs = det_res.boxes.conf.cpu().numpy()
        classes = det_res.boxes.cls.cpu().numpy()
        for box, conf, cls in zip(boxes, confs, classes):
            if conf < 0.5: continue
            x1d, y1d, x2d, y2d = map(int, box)
            track_inputs.append(([x1d, y1d, x2d-x1d, y2d-y1d], conf, int(cls)))

    tracks = tracker.update_tracks(track_inputs, frame=frame)
    for track in tracks:
        if not track.is_confirmed(): continue
        tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
        cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 255, 255), 2)
        cv2.putText(overlay, f"ID:{track.track_id}", (tx1, ty1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # -------- Intensity Graph Overlay (TOP of video) --------
    if len(intensity_history) > 2:
        graph_h, graph_w = 150, 500  # make graph wider
        graph = np.ones((graph_h, graph_w, 3), dtype=np.uint8) * 255  # white bg

        # Show last `graph_w-40` values (scrolling effect)
        vals_to_plot = list(intensity_history)[- (graph_w - 40):]
        norm_vals = [int(min(v,150)/150 * (graph_h-40)) for v in vals_to_plot]

        # Draw axis
        cv2.line(graph, (30, graph_h-20), (graph_w-10, graph_h-20), (0,0,0), 1) # x-axis
        cv2.line(graph, (30, 10), (30, graph_h-20), (0,0,0), 1) # y-axis

        # Labels
        cv2.putText(graph, "Intensity (0-150)", (35, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        # Draw curve
        for i in range(1, len(norm_vals)):
            x1 = 30 + (i-1)
            y1 = graph_h-20 - norm_vals[i-1]
            x2 = 30 + i
            y2 = graph_h-20 - norm_vals[i]
            if x2 < graph_w-10:
                cv2.line(graph, (x1,y1), (x2,y2), (0,0,0), 1)

        # Add ticks for 0–150
        for y in range(0, 151, 30):
            y_pos = graph_h-20 - int(y/150 * (graph_h-40))
            cv2.line(graph, (25,y_pos), (30,y_pos), (0,0,0), 1)
            cv2.putText(graph, str(y), (2,y_pos+3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)

        # Place graph at TOP LEFT
        overlay[0:graph_h, 0:graph_w] = graph

    # -------- Show & Save --------
    out.write(overlay)
    cv2.imshow("Integrated Output", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); out.release(); cv2.destroyAllWindows()
