# NeuralStyleTransfer

**COMPANY** : CODTECH IT SOLUTIONS

**NAME** : SIA SHAH

**INTERN ID** : CT08JOP

**DOMAIN NAME** : ARTIFICIAL INTELLIGENCE

**BATCH DURATION** : JANUARY 6TH,2025 TO FEBRUARY 6TH,2025

**MENTOR NAME** : NEELA SANTOSH

# Description of Neural Style Transfer 
**Neural Style Transfer (NST)** is a technique in deep learning that applies the artistic style of one image (the "style image") to the content of another image (the "content image"). It creates a blended image that preserves the structural elements of the content image while adopting the textures, colors, and patterns of the style image.

### Key Concepts:
1. **Content**: Represents the high-level features (shapes, objects, structure) of the content image.
2. **Style**: Captures the artistic elements (textures, colors, brush strokes) of the style image using patterns of correlations between features (Gram matrices).
3. **Deep Learning Models**: Pre-trained convolutional neural networks (e.g., VGG19) extract features from images at different layers. Deeper layers capture content, while shallow layers capture style.

### How NST Works:
- **Feature Extraction**: A pre-trained network like VGG19 extracts features from the content and style images.
- **Loss Functions**:
  - **Content Loss**: Ensures the target image resembles the content image in structure.
  - **Style Loss**: Matches the style of the target image to the style image by aligning their Gram matrices.
- **Optimization**: The target image (initialized as the content image or random noise) is iteratively updated to minimize the combined loss, producing the stylized output.

### Applications:
- Artistic photo editing
- Image enhancement for design and advertising
- Creative content generation for games, films, and VR

# Content Image 
![Image](https://github.com/user-attachments/assets/7488745f-25ec-473e-b6cb-86779a0a519b)

# Style Image 
![Image](https://github.com/user-attachments/assets/5c8dd32b-530d-4151-ba73-ad6a260cfd58)
