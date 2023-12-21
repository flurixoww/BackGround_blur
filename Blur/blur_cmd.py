import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
import numpy as np 
from PIL import Image
path = input("Insert path to your picture: ")
image_cv2 = cv2.imread(path)
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
blurred_background = cv2.GaussianBlur(orig_image, (21, 21), 0)
mask_inv = cv2.bitwise_not(mask)
person = cv2.bitwise_and(orig_image, orig_image, mask=mask)
background = cv2.bitwise_and(blurred_background, blurred_background, mask=mask_inv)
final_image = cv2.add(person, background)
final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('blurred_background_image.jpg', final_image)