import torch
path=r'model_data/facenet_mobilenet.pth'
model=torch.load(path,map_location='cpu')
print(type(model))
print(len(model))
for k in model:
    print(k,model[k])