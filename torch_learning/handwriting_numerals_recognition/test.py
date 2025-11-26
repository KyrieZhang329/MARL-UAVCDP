import torch
from torchvision import transforms
from PIL import Image
from Net import Net
import os

state_dict = torch.load('./final_model2.pth')
model = Net()
model.load_state_dict(state_dict)
# 识别单张图片
img = Image.open("./numerals/yh/1.jpg").convert('L')
transform1 = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
img_tensor = transform1(img)  # transforms.ToTensor()已经返回了tensor
img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度

model.eval()
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    print(f'这是: {predicted.item()}')


# 批量识别
num_1_folder = "./numerals/num_1"
num_1_images = []
for filename in os.listdir(num_1_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        num_1_images.append(os.path.join(num_1_folder, filename))

for img_path in num_1_images:
    img = Image.open(img_path).convert('L')
    transform1 = transforms.Compose([
        transforms.Resize((28, 28)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))  
    ])
    img_tensor = transform1(img)  # transforms.ToTensor()已经返回了tensor
    img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        print(f'{os.path.basename(img_path)} 识别结果: {predicted.item()}')


# num_2_folder = "./numerals/num_2"
# num_2_images = []
# for filename in os.listdir(num_2_folder):
#     if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
#         num_2_images.append(os.path.join(num_2_folder, filename))

# for img_path in num_2_images:
#     img = Image.open(img_path).convert('L')
#     transform1 = transforms.Compose([
#         transforms.Resize((28, 28)), 
#         transforms.ToTensor(),  
#         transforms.Normalize((0.5,), (0.5,))  
#     ])
#     img_tensor = transform1(img)  # transforms.ToTensor()已经返回了tensor
#     img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
    
#     model.eval()
#     with torch.no_grad():
#         output = model(img_tensor)
#         _, predicted = torch.max(output, 1)
#         print(f'{os.path.basename(img_path)} 识别结果: {predicted.item()}')