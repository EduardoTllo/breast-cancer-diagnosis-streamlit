import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F

# --- MACENKO NORMALIZATION ---
class MacenkoNormalizer:
    """
    Implementación de normalización de tinción Macenko.
    Referencia: Macenko et al. (2009)
    """
    def __init__(self):
        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def normalize(self, img, Io=240, alpha=1, beta=0.15):
        """
        Normaliza una imagen RGB.
        img: numpy array (H, W, 3) en rango [0, 255]
        """
        h, w, c = img.shape
        img = img.reshape((-1, 3))

        # Calcular densidad óptica (OD)
        OD = -np.log((img.astype(np.float64) + 1) / Io)

        # Eliminar datos transparentes
        ODhat = OD[~np.any(OD < beta, axis=1)]
        if len(ODhat) == 0:
            return img.reshape(h, w, c)

        # Calcular vectores propios
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        
        # Proyectar en el plano de los dos vectores propios principales
        That = ODhat.dot(eigvecs[:, 1:3])

        # Encontrar vectores de tinción extremos
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)

        vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T

        # Concentraciones de tinción
        Y = np.reshape(OD, (-1, 3)).T
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]

        # Normalizar concentraciones
        maxC = np.percentile(C, 99, axis=1)
        tmp = np.divide(maxC, self.maxCRef)
        C2 = np.divide(C, tmp[:, np.newaxis])

        # Reconstruir imagen
        Inorm = np.multiply(Io, np.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 255
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

        return Inorm

# Instancia global
macenko = MacenkoNormalizer()

# --- MODEL LOADING ---
def build_resnet18(num_classes=2):
    """Construye la arquitectura ResNet18 modificada."""
    # No necesitamos pesos preentrenados aquí porque cargaremos nuestro checkpoint
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_model(model_path, device):
    """Carga el modelo desde el path especificado."""
    model = build_resnet18(num_classes=2)
    
    try:
        # Cargar state_dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Manejar si el checkpoint tiene una key 'model' o es el dict directo
        if 'model' in state_dict:
            state_dict = state_dict['model']
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error cargando el modelo: {e}")

# --- PREPROCESSING ---
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def preprocess_image(image_file, use_macenko=False):
    """
    Preprocesa la imagen cargada para el modelo.
    image_file: Archivo subido por st.file_uploader
    Returns: 
        tensor: (1, 3, 224, 224) listo para inferencia
        original_image: PIL Image para visualización
    """
    # Leer imagen
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Macenko opcional
    if use_macenko:
        try:
            img = macenko.normalize(img)
        except Exception:
            pass # Fallback silencioso
            
    # Convertir a PIL para transforms
    img_pil = Image.fromarray(img)
    
    # Transformaciones de validación/test
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0) # Batch dim
    
    return img_tensor, img_pil

# --- GRAD-CAM ---
class GradCAM:
    """
    Implementación simple de Grad-CAM para ResNet.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # Backward
        score = output[0, class_idx]
        score.backward()
        
        # Generar mapa
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling de gradientes
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Combinación lineal
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalizar min-max
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.detach().cpu().numpy()[0, 0], output

def overlay_cam(img_pil, cam_mask, alpha=0.5):
    """Superpone el mapa de calor en la imagen original."""
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    
    # Redimensionar máscara al tamaño de la imagen
    heatmap = cv2.resize(cam_mask, (w, h))
    
    # Colorear
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superponer
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlay)
