import torch




import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
import numpy as np 
from PIL import Image
image_cv2 = cv2.imread(r"D:\Programming\b_blur\1231233.jpg")
orig_image = image_cv2.copy()
image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
image_PIL = Image.fromarray(image_cv2)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = transform(image_PIL).unsqueeze(0)
model = deeplabv3_resnet50(pretrained=True)
model.eval()
with torch.no_grad():
    output = model(image)['out'][0]
output_predictions = output.argmax(0)
mask = output_predictions == output_predictions.max()
mask = output_predictions.cpu().numpy()  # Move tensor to CPU and convert to NumPy array
mask = (mask == mask.max()).astype(np.uint8) * 255  # Create binary mask for the person
mask = cv2.resize(mask, (orig_image.shape[1], orig_image.shape[0]))
person = cv2.bitwise_and(orig_image, orig_image, mask=mask)
background = np.zeros_like(orig_image)
final_image = cv2.add(person, background)
cv2.imwrite('removed.jpg', final_image)