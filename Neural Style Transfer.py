#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and preprocess image
def load_image(image_path, max_size=400):
    image = Image.open(image_path)
    size = max(image.size)
    if size > max_size:
        ratio = max_size / float(size)
        new_size = tuple([int(dim * ratio) for dim in image.size])
        image = image.resize(new_size, Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function to display images
def display_image(tensor):
    image = tensor.squeeze(0).detach().cpu()
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Load content and style images
content_image_path = "D:\Siya\Downloads\Premium Photo _ Scenic Photos of Mountains Forests Oceans Sunsets and Natural Settings.jpeg"  # Replace with your content image path
style_image_path = "D:\Siya\Downloads\Van Gogh painting, European art, Night landscape, Stars, Starry night Vincent Van Gogh FINE ART PRINT, Impressionism, Wall art, Art posters.jpeg"      # Replace with your style image path

content_image = load_image(content_image_path).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
style_image = load_image(style_image_path).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Load pre-trained VGG19 model
cnn = models.vgg19(pretrained=True).features.eval().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Freeze the model's parameters
for param in cnn.parameters():
    param.requires_grad = False

# Extract features using VGG19
class StyleTransferModel(nn.Module):
    def __init__(self, cnn, content_layers, style_layers):
        super(StyleTransferModel, self).__init__()
        self.cnn = cnn
        self.content_layers = content_layers
        self.style_layers = style_layers

    def forward(self, x):
        content_features = []
        style_features = []
        for name, layer in self.cnn._modules.items():
            x = layer(x)
            if name in self.content_layers:
                content_features.append(x)
            if name in self.style_layers:
                style_features.append(x)
        return content_features, style_features

# Define layers for content and style extraction
content_layers = ['21']  # Use features from layer 21 of VGG19
style_layers = ['0', '5', '10', '19', '28']  # Use features from layers 0, 5, 10, 19, and 28

# Initialize model
model = StyleTransferModel(cnn, content_layers, style_layers).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Function to compute the Gram matrix for style loss
def gram_matrix(x):
    batch_size, channel, height, width = x.size()
    features = x.view(batch_size * channel, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * channel * height * width)

# Function to compute content loss
def compute_content_loss(content, target):
    return torch.mean((content - target) ** 2)

# Function to compute style loss
def compute_style_loss(style, target):
    gram_style = gram_matrix(style)
    gram_target = gram_matrix(target)
    return torch.mean((gram_style - gram_target) ** 2)

# Extract content and style features for the images
content_features, _ = model(content_image)
_, style_features = model(style_image)

# Initialize the target image (copy of content image)
target = content_image.clone().requires_grad_(True).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Define optimizer
optimizer = optim.Adam([target], lr=0.01)

# Run optimization
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Get the features of the target image
    target_content_features, target_style_features = model(target)

    # Compute content and style loss
    content_loss = compute_content_loss(content_features[0], target_content_features[0])
    style_loss = 0
    for i in range(len(style_features)):
        style_loss += compute_style_loss(style_features[i], target_style_features[i])

    # Total loss
    total_loss = content_loss + style_loss * 100
    total_loss.backward()

    # Update the target image
    optimizer.step()

    # Display the styled image every 50 epochs
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Total Loss: {total_loss.item()}')
        display_image(target)

# Final result after optimization
display_image(target)


# In[ ]:




