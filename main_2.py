import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
from ultralytics import YOLO
from collections import defaultdict

# 1. Khởi tạo Mô hình
model = YOLO('yolo11l.pt')
class_list = model.names
video_path = './test videos/vehicle-counting.mp4'
cap = cv2.VideoCapture(video_path)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Kích thước hiển thị
DISPLAY_W, DISPLAY_H = 1280, 720
target_line_y_4k = 1200 

out = cv2.VideoWriter('output_final.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))

class_counts = {"In": defaultdict(int), "Out": defaultdict(int)}

track_history = defaultdict(list) 
track_colors = {} 
crossed_ids = set() 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # CẬP NHẬT 1: Dùng custom_tracker.yaml và tăng conf lên 0.50
    results = model.track(
        frame, 
        persist=True, 
        tracker="custom_tracker.yaml", 
        classes=[1,2,3,5,6,7]
    )
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        current_frame_ids = set(track_ids)

        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            # CẬP NHẬT 2: Lọc bỏ các bounding box quá nhỏ (xe ở quá xa hoặc nhiễu)
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width < 40 or box_height < 40:
                continue # Bỏ qua frame này, không xử lý cấp ID hay vẽ vời gì cả

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = class_list[class_idx]
            conf_percentage = int(conf * 100)

            # Ghi nhận lịch sử di chuyển (Lưu ngắn 8 frame để phản ứng nhanh khi vừa hết che khuất)
            history = track_history[track_id]
            history.append(cy)
            if len(history) > 8: 
                history.pop(0)

            # LOGIC ĐỐM KHẮC PHỤC CHE KHUẤT: Dựa trên Vector xu hướng thực tế
            if track_id not in crossed_ids and len(history) >= 3:
                start_cy = history[0]
                
                # Điều kiện kích hoạt đếm: Tâm xe (cy) phải cắt qua vạch đỏ trong hành trình của nó
                # Kiểm tra xe đi từ trên xuống (Chiều IN)
                if start_cy <= target_line_y_4k and cy >= target_line_y_4k:
                    if cy > start_cy: # Đảm bảo xe thực sự di chuyển hướng xuống
                        class_counts["In"][class_name] += 1
                        track_colors[track_id] = (0, 255, 0) # Màu Xanh lá
                        crossed_ids.add(track_id)
                
                # Kiểm tra xe đi từ dưới lên (Chiều OUT)
                elif start_cy >= target_line_y_4k and cy <= target_line_y_4k:
                    if cy < start_cy: # Đảm bảo xe thực sự di chuyển hướng lên
                        class_counts["Out"][class_name] += 1
                        track_colors[track_id] = (0, 255, 255) # Màu Vàng
                        crossed_ids.add(track_id)

            # Chọn màu hiển thị (Mặc định trắng nếu chưa qua vạch)
            color = track_colors.get(track_id, (255, 255, 255))

            # Vẽ đồ họa lên frame gốc 4K
            cv2.circle(frame, (cx, cy), 10, color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id} {class_name} {conf_percentage}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Dọn dẹp lịch sử an toàn tránh rác dữ liệu
        all_tracked_ids = list(track_history.keys())
        for tid in all_tracked_ids:
            if tid not in current_frame_ids:
                if len(track_history[tid]) > 0:
                    track_history[tid].pop(0)
                if len(track_history[tid]) == 0:
                    del track_history[tid]

    # Vẽ đường ranh giới đếm xe
    cv2.line(frame, (0, target_line_y_4k), (orig_w, target_line_y_4k), (0, 0, 255), 5)
    display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    
    # Giao diện hiển thị thống kê In/Out ở 2 góc màn hình
    cv2.putText(display_frame, "Entrance:", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    y_in = 90
    for name, count in class_counts["In"].items():
        cv2.putText(display_frame, f"{name}: {count}", (20, y_in), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_in += 40

    cv2.putText(display_frame, "Exit:", (1110, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    y_out = 90
    for name, count in class_counts["Out"].items():
        cv2.putText(display_frame, f"{name}: {count}", (1110, y_out), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_out += 40

    out.write(frame)
    cv2.imshow("YOLO Tracking 2-Way", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()