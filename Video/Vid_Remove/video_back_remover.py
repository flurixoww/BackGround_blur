import torch
import cv2  
from torchvision import models, transforms
import numpy as np
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
def segment(frame, orig_dims):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    mask = output_predictions.byte().cpu().numpy()
    mask = cv2.resize(mask, (orig_dims[1], orig_dims[0]))
    return mask
cap = cv2.VideoCapture(r'D:\Programming\b_blur\images&videos\vid.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
orig_dims = (frame_height, frame_width)
out = cv2.VideoWriter('output1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    mask = segment(frame, orig_dims)
    frame[mask == 0] = [0, 0, 0]
    out.write(frame)
cap.release()
out.release()
cv2.destroyAllWindows()