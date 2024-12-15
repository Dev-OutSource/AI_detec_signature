import requests

def send_image_to_api(image_path):
    # Địa chỉ API mà bạn muốn gửi yêu cầu đến
    api_url = "http://127.0.0.1:8000/process-image"
    
    # Mở file hình ảnh để gửi qua API
    with open(image_path, "rb") as image_file:
        # Tạo dữ liệu yêu cầu
        files = {"file": ("image.jpg", image_file, "image/jpeg")}
        
        # Gửi yêu cầu POST với hình ảnh
        response = requests.post(api_url, files=files)
        
        # Kiểm tra mã trạng thái của phản hồi
        if response.status_code == 200:
            # Nếu thành công, in ra kết quả trả về từ API
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")

# Đường dẫn đến hình ảnh bạn muốn gửi
image_path = "test_dataset/di-cham.jpg"

# Gửi hình ảnh đến API
result = send_image_to_api(image_path)
print(result['traffics'])
