import os
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Kiểm tra xem thư mục có tồn tại không
data_path = "Data_Blood_Nude/noblood1"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Thư mục {data_path} không tồn tại! Kiểm tra lại đường dẫn.")

# Định nghĩa các biến đổi ảnh
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load dữ liệu
dataset_no_blood = ImageFolder(data_path, transform=transform)
print(f"Đã tải thành công {len(dataset_no_blood)} ảnh từ {data_path}")
