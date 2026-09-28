import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
from ultralytics import YOLO
from collections import defaultdict

# 1. Khởi tạo Mô hình
model = YOLO('yolo11l.pt')
class_list = model.names
video_path = './test videos/test_1.mp4'
cap = cv2.VideoCapture(video_path)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
DISPLAY_W, DISPLAY_H = 1280, 720

# Vị trí vạch đếm (điều chỉnh)
# test_1.mp4: 0.65
# test_2.mp4: 0.60
# test_3.mp4: 0.40
# test_4.mp4: 0.56

LINE_RATIO = 0.65
YLine = int(orig_h * LINE_RATIO)  

NIGHT_MODE = True  

if NIGHT_MODE:
    CONF_THRESH = 0.25  
    IOU_THRESH = 0.65   
    MIN_BOX_SIZE = 30   
else:
    CONF_THRESH = 0.40  # Giảm ngưỡng tự tin để phát hiện nhiều phương tiện hơn
    IOU_THRESH = 0.50   # Giảm IoU để tránh gộp các phương tiện đi sát nhau
    MIN_BOX_SIZE = 15   # Giảm kích thước tối thiểu để phát hiện phương tiện nhỏ

out = cv2.VideoWriter('output_final.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))

class_counts = {"In": defaultdict(int), "Out": defaultdict(int)}
track_history = defaultdict(list) 
crossed_ids = set() 
track_colors = {} 
display_id_map = {}
next_display_id = 1

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        processed_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30) if NIGHT_MODE else frame

        results = model.track(
            processed_frame, persist=True, tracker="custom_tracker.yaml", 
            classes=[1,2,3,5,6,7], conf=CONF_THRESH, iou=IOU_THRESH, imgsz=640  
        )
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()

            current_frame_ids = set(track_ids)

            for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                if (x2 - x1) < MIN_BOX_SIZE and (y2 - y1) < MIN_BOX_SIZE: continue 

                if track_id not in display_id_map:
                    display_id_map[track_id] = next_display_id
                    next_display_id += 1
                ui_id = display_id_map[track_id] 

                Cx, Cy = (x1 + x2) // 2, (y1 + y2) // 2
                class_name = class_list[class_idx]
                conf_percentage = int(conf * 100)

                history = track_history[track_id]
                history.append(Cy)
                if len(history) > 30: 
                    history.pop(0)

                if track_id not in crossed_ids and len(history) >= 10:
                    start_cy = history[0] 
                    
                    if start_cy <= YLine and Cy >= YLine:
                        if Cy > start_cy + 10: 
                            class_counts["In"][class_name] += 1
                            track_colors[track_id] = (0, 255, 0) 
                            crossed_ids.add(track_id)
                            
                    elif start_cy >= YLine and Cy <= YLine:
                        if Cy < start_cy - 10: 
                            class_counts["Out"][class_name] += 1
                            track_colors[track_id] = (0, 255, 255) 
                            crossed_ids.add(track_id)

                color = track_colors.get(track_id, (255, 255, 255))
                cv2.circle(frame, (Cx, Cy), 10, color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {ui_id} {class_name} {conf_percentage}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.line(frame, (0, YLine), (orig_w, YLine), (0, 0, 255), 4) 
        
        display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        
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

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

except KeyboardInterrupt:
    pass
finally:
    print("\n" + "="*55)
    print("BÁO CÁO THỐNG KÊ LƯU LƯỢNG GIAO THÔNG (SYSTEM SUMMARY)")
    print("="*55)
    print(f"{'Phân loại':<15} | {'Vào (IN)':<10} | {'Ra (OUT)':<10} | {'Tổng cộng':<10}")
    print("-" * 55)

    detected_classes = set(class_counts["In"].keys()).union(set(class_counts["Out"].keys()))

    total_in, total_out = 0, 0
    for name in detected_classes:
        c_in = class_counts["In"][name]
        c_out = class_counts["Out"][name]
        c_total = c_in + c_out
        total_in += c_in
        total_out += c_out
        print(f"{name:<15} | {c_in:<10} | {c_out:<10} | {c_total:<10}")

    print("-" * 55)
    print(f"{'TỔNG TẤT CẢ':<15} | {total_in:<10} | {total_out:<10} | {total_in + total_out:<10}")
    print("="*55)

    print("[*] Đang giải phóng tài nguyên bộ nhớ...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("[*] Hoàn tất!")