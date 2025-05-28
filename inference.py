import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🏁 Pipeline Completo - Dispositivo: {device}')

# ==================== LIGHTWEIGHT U-NET ====================

class LightChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(LightChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class EfficientDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(EfficientDoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)  # Reduzido de 0.05
        self.use_attention = use_attention
        if use_attention:
            self.attention = LightChannelAttention(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        if self.use_attention:
            x = self.attention(x)
        return x

class LightweightUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(LightweightUNet, self).__init__()
        
        # Encoder
        self.enc1 = EfficientDoubleConv(in_channels, 32, use_attention=False)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = EfficientDoubleConv(32, 64, use_attention=True)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = EfficientDoubleConv(64, 128, use_attention=True)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = EfficientDoubleConv(128, 256, use_attention=True)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = EfficientDoubleConv(256, 512, use_attention=True)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = EfficientDoubleConv(512, 256, use_attention=False)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = EfficientDoubleConv(256, 128, use_attention=False)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = EfficientDoubleConv(128, 64, use_attention=False)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = EfficientDoubleConv(64, 32, use_attention=False)

        # Saída
        self.final_conv = nn.Conv2d(32, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1_pool = self.pool1(e1)
        e2 = self.enc2(e1_pool)
        e2_pool = self.pool2(e2)
        e3 = self.enc3(e2_pool)
        e3_pool = self.pool3(e3)
        e4 = self.enc4(e3_pool)
        e4_pool = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(e4_pool)

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        output = self.sigmoid(self.final_conv(d1))
        return output

# ==================== LOSS FUNCTION COMBINADA ====================

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        
    def dice_loss(self, pred, target, smooth=1e-6):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

# ==================== DATASET SIMPLIFICADO ====================

class RoadDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=(256, 256), augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def simple_augmentation(self, image, mask):
        """Augmentação conservadora"""
        # Flip horizontal apenas
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # Variações de brilho suaves
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # Rotações muito pequenas
        if np.random.random() > 0.7:
            angle = np.random.uniform(-3, 3)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
        
        return image, mask

    def __getitem__(self, idx):
        # Carregar imagem
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)

        # Carregar máscara
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)

        # Augmentação conservadora
        if self.augment:
            image, mask = self.simple_augmentation(image, mask)

        # Normalização consistente
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Converter para tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

# ==================== EARLY STOPPING ====================

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.wait = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                print(f'✅ Validation loss melhorou para {val_loss:.6f}')
        else:
            self.wait += 1
            if self.verbose:
                print(f'⚠️ Sem melhoria há {self.wait} épocas')

        if self.wait >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print('🔄 Restaurados melhores pesos')

# ==================== FUNÇÕES DE DEBUG ====================

def debug_dataset(train_loader, num_batches=3):
    """Verificar se os dados estão corretos"""
    print("🔍 Verificando dataset...")
    
    for i, (images, masks) in enumerate(train_loader):
        if i >= num_batches:
            break
            
        print(f"Batch {i}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images range: {images.min():.4f} - {images.max():.4f}")
        print(f"  Masks shape: {masks.shape}")
        print(f"  Masks range: {masks.min():.4f} - {masks.max():.4f}")
        print(f"  Masks unique values: {torch.unique(masks)}")
        
        # Verificar se há máscaras com conteúdo
        mask_sum = masks.sum(dim=[2,3])
        print(f"  Masks with content: {(mask_sum > 0).sum().item()}/{masks.shape[0]}")
        
        # Visualizar algumas amostras
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for j in range(min(4, images.shape[0])):
            img = images[j].permute(1, 2, 0).numpy()
            mask = masks[j].squeeze().numpy()
            
            axes[0, j].imshow(img)
            axes[0, j].set_title(f'Image {j}')
            axes[0, j].axis('off')
            
            axes[1, j].imshow(mask, cmap='gray')
            axes[1, j].set_title(f'Mask {j} (sum: {mask.sum():.0f})')
            axes[1, j].axis('off')
        
        plt.tight_layout()
        plt.show()
        break

def create_prediction_stack(model, val_loader, device, num_samples=5, epoch=0):
    """Criar visualizações das predições"""
    model.eval()
    os.makedirs('predictions', exist_ok=True)
    
    images_list = []
    masks_list = []
    predictions_list = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            if len(images_list) >= num_samples:
                break
            
            images, masks = images.to(device), masks.to(device)
            predictions = model(images)
            
            for i in range(min(images.shape[0], num_samples - len(images_list))):
                img = images[i].cpu().permute(1, 2, 0).numpy()
                mask = masks[i].cpu().squeeze().numpy()
                pred = predictions[i].cpu().squeeze().numpy()
                
                images_list.append(img)
                masks_list.append(mask)
                predictions_list.append(pred)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    fig.suptitle(f'🏁 Lightweight U-Net - Época {epoch}', fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        axes[i, 0].imshow(images_list[i])
        axes[i, 0].set_title(f'🏁 Imagem {i+1}', fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masks_list[i], cmap='gray')
        axes[i, 1].set_title(f'🎯 Ground Truth {i+1}', fontweight='bold')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(predictions_list[i], cmap='gray')
        axes[i, 2].set_title(f'🤖 Predição {i+1}', fontweight='bold')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = images_list[i].copy()
        line_mask = predictions_list[i] > 0.5
        overlay[line_mask] = overlay[line_mask] * 0.7 + np.array([0, 1, 0]) * 0.3
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f'🌈 Overlay {i+1}', fontweight='bold')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(f'predictions/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# ==================== PREPARAÇÃO DOS DADOS ====================

def prepare_data(images_path, masks_path, img_size=(256, 256), test_size=0.2):
    """Preparar dados para treino"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(images_path, ext)))
        image_files.extend(glob(os.path.join(images_path, ext.upper())))

    mask_files = []
    for ext in image_extensions:
        mask_files.extend(glob(os.path.join(masks_path, ext)))
        mask_files.extend(glob(os.path.join(masks_path, ext.upper())))

    image_files = sorted(image_files)
    mask_files = sorted(mask_files)

    print(f"🏁 Encontradas {len(image_files)} imagens e {len(mask_files)} máscaras")

    X_train, X_val, y_train, y_val = train_test_split(
        image_files, mask_files, test_size=test_size, random_state=42
    )

    train_dataset = RoadDataset(X_train, y_train, img_size, augment=True)
    val_dataset = RoadDataset(X_val, y_val, img_size, augment=False)

    return train_dataset, val_dataset

# ==================== TREINO MELHORADO ====================

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, patience=15, visualization_interval=5):
    """Treinar o modelo com configurações otimizadas"""
    
    # Loss function combinada
    criterion = CombinedLoss(alpha=0.7)
    
    # Optimizer com learning rate mais alto
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Scheduler menos agressivo
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.7, verbose=True, min_lr=1e-6
    )
    
    # Early stopping menos agressivo
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_losses = []
    val_losses = []

    print(f"🚀 Iniciando treino com learning rate {learning_rate}...")

    for epoch in range(num_epochs):
        # Treino
        model.train()
        train_loss = 0.0
        batch_losses = []
        
        train_bar = tqdm(train_loader, desc=f'Época {epoch+1:02d}/{num_epochs} [TREINO]')
        
        for batch_idx, (images, masks) in enumerate(train_bar):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Verificar gradientes
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            optimizer.step()

            batch_losses.append(loss.item())
            train_loss += loss.item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Grad': f'{total_norm:.2f}'
            })

        # Validação
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Época {epoch+1:02d}/{num_epochs} [VALID]')
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f'📈 Época {epoch+1:02d}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}')
        print(f'   Output range: {outputs.min():.4f} - {outputs.max():.4f}')
        print(f'   Grad norm: {total_norm:.4f}')

        # Visualizações periódicas
        if (epoch + 1) % visualization_interval == 0:
            print(f"🎨 Criando visualizações...")
            create_prediction_stack(model, val_loader, device, num_samples=5, epoch=epoch+1)

        # Early stopping e scheduler
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping na época {epoch+1}")
            break

        scheduler.step(avg_val_loss)

    return train_losses, val_losses, early_stopping.best_loss

# ==================== PROCESSAMENTO DE VÍDEO SIMPLIFICADO ====================

class VideoLineDetector:
    def __init__(self, model_path, img_size=(256, 256)):
        self.img_size = img_size
        self.model = LightweightUNet()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        print(f"✅ Modelo carregado: {model_path}")

    def preprocess_frame(self, frame):
        """Pré-processamento consistente com o treino"""
        frame_resized = cv2.resize(frame, self.img_size)
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        return frame_normalized

    def process_frame(self, frame):
        """Processar um frame do vídeo"""
        processed_frame = self.preprocess_frame(frame)
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(processed_frame).permute(2, 0, 1).unsqueeze(0)
            input_tensor = input_tensor.to(device)
            mask_pred = self.model(input_tensor)
            mask_np = mask_pred.cpu().squeeze().numpy()
        
        return mask_np

    def create_visualization(self, original_frame, mask, threshold=0.3):
        """Criar visualização com overlay"""
        h, w = original_frame.shape[:2]
        mask_resized = cv2.resize(mask, (w, h))
        
        overlay = original_frame.copy()
        line_mask = mask_resized > threshold
        overlay[line_mask] = overlay[line_mask] * 0.6 + np.array([0, 255, 0]) * 0.4
        
        # Adicionar informações
        cv2.putText(overlay, f'Threshold: {threshold:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f'Max: {mask.max():.3f}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return overlay

# ==================== SCRIPT PRINCIPAL ====================

def main():
    print("=" * 70)
    print("🏁 PIPELINE CORRIGIDO - TREINO E TESTE")
    print("=" * 70)

    # Configurações
    IMAGES_PATH = "../images"
    MASKS_PATH = "../masks"
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3  # Aumentado
    PATIENCE = 15  # Aumentado
    VISUALIZATION_INTERVAL = 5
    MODEL_PATH = "lightweight_unet_corrected.pth"

    # Escolher modo
    mode = input("Escolhe o modo:\n1 - Treinar modelo\n2 - Debug dataset\n3 - Ambos\nOpção: ")

    if mode in ['2', '3']:
        print("\n🔍 VERIFICANDO DATASET...")
        train_dataset, val_dataset = prepare_data(IMAGES_PATH, MASKS_PATH, IMG_SIZE)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        debug_dataset(train_loader)

    if mode in ['1', '3']:
        print("\n🚀 INICIANDO TREINO...")
        
        # Preparar dados
        train_dataset, val_dataset = prepare_data(IMAGES_PATH, MASKS_PATH, IMG_SIZE)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        print(f"📚 Dataset treino: {len(train_dataset)} amostras")
        print(f"📚 Dataset validação: {len(val_dataset)} amostras")

        # Criar modelo
        model = LightweightUNet().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🏗️ Modelo criado: {total_params:,} parâmetros")

        # Treinar
        train_losses, val_losses, best_loss = train_model(
            model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, PATIENCE, VISUALIZATION_INTERVAL
        )

        # Salvar modelo
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Modelo guardado: {MODEL_PATH}")

        # Plotar resultados
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Treino', color='blue', linewidth=2)
        plt.plot(val_losses, label='Validação', color='red', linewidth=2)
        plt.title('🏁 Loss Curves', fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.text(0.1, 0.5, f"""
🏁 RESULTADOS FINAIS

📊 Melhor Val Loss: {best_loss:.6f}
📏 Parâmetros: {total_params:,}
🎯 Learning Rate: {LEARNING_RATE}
""", fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('training_results_corrected.png', dpi=150, bbox_inches='tight')
        plt.show()

    print("\n✅ Pipeline finalizado!")

if __name__ == "__main__":
    main()

