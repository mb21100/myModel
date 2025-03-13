import os
import cv2

# 목표 해상도 (width, height)
target_width = 3840
target_height = 2160

# 메인 폴더 경로 (여기에는 24개의 서브폴더가 있다고 가정)
main_folder = "/home/mb21100/data/SR4KIQA"  

# 처리할 이미지 파일 확장자 목록 (소문자로)
allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

# 메인 폴더 내 모든 파일을 재귀적으로 순회
for root, dirs, files in os.walk(main_folder):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in allowed_extensions:
            file_path = os.path.join(root, file)
            # 이미지 읽기
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"이미지를 읽을 수 없습니다: {file_path}")
                continue
            h, w = img.shape[:2]
            # 해상도가 목표와 다르면 리사이즈
            if (w, h) != (target_width, target_height):
                resized_img = cv2.resize(img, (target_width, target_height))
                # 이미지 저장 (원본 파일을 덮어씁니다)
                cv2.imwrite(file_path, resized_img)
                print(f"Resized {file_path}: {w}x{h} -> {target_width}x{target_height}")
