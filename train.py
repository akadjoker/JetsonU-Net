import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🏁 Sistema Otimizado de Detecção de Linhas - Dispositivo: {device}')

# ==================== LIGHTWEIGHT ATTENTION ====================
class LightChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):  # Redução maior
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

# ==================== EFFICIENT CONV BLOCKS ====================
class EfficientDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(EfficientDoubleConv, self).__init__()
 
        max_groups = min(in_channels, out_channels, 8)
        groups = 1
        
        # Encontrar o maior grupo válido
        for g in range(max_groups, 0, -1):
            if in_channels % g == 0 and out_channels % g == 0:
                groups = g
                break
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, 
                              groups=groups, bias=False)  # Usar grupos calculados
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.05)
        
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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, 
                                  groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(self.bn(x))
        return x

# ==================== LIGHTWEIGHT U-NET ====================
class LightweightUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(LightweightUNet, self).__init__()
        
        # Encoder com menos filtros
        self.enc1 = EfficientDoubleConv(in_channels, 32, use_attention=False)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = EfficientDoubleConv(32, 64, use_attention=True)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = EfficientDoubleConv(64, 128, use_attention=True)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = EfficientDoubleConv(128, 256, use_attention=True)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck mais leve
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(256, 512),
            DepthwiseSeparableConv(512, 512),
            LightChannelAttention(512)
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = EfficientDoubleConv(512, 256, use_attention=False)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = EfficientDoubleConv(256, 128, use_attention=False)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = EfficientDoubleConv(128, 64, use_attention=False)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = EfficientDoubleConv(64, 32, use_attention=False)
        
        # Saída final simples
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
        
        # Saída
        output = self.sigmoid(self.final_conv(d1))
        return output

# ==================== DATASET   ====================
class RoadDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=(256, 256), augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment
    
    def __len__(self):
        return len(self.image_paths)
    
    def augment_data(self, image, mask):
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        return image, mask
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)
        
        if self.augment:
            image, mask = self.augment_data(image, mask)
        
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

# ==================== EARLY STOPPING ====================
class EarlyStopping:
    def __init__(self, patience=12, min_delta=0, restore_best_weights=True, verbose=True):
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

# ==================== VISUALIZAÇÃO ====================
def create_prediction_stack(model, val_loader, device, num_samples=5, epoch=0):
    model.eval()
    os.makedirs('lightweight_predictions', exist_ok=True)
    
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
        axes[i, 0].set_title(f'🏁 Pista {i+1}', fontweight='bold')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masks_list[i], cmap='gray')
        axes[i, 1].set_title(f'🎯 Real {i+1}', fontweight='bold')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(predictions_list[i], cmap='gray')
        axes[i, 2].set_title(f'🤖 Predição {i+1}', fontweight='bold')
        axes[i, 2].axis('off')
        
        overlay = images_list[i].copy()
        line_mask = predictions_list[i] > 0.5
        overlay[line_mask] = overlay[line_mask] * 0.7 + np.array([1, 1, 0]) * 0.3
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f'🌈 Overlay {i+1}', fontweight='bold')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'lightweight_predictions/lightweight_epoch_{epoch:03d}.png', 
                dpi=150, bbox_inches='tight')
    #plt.show()
    plt.close()

# ==================== FUNÇÕES AUXILIARES ====================
def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def prepare_data(images_path, masks_path, img_size=(256, 256), test_size=0.2):
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

# ==================== TREINAMENTO ====================
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, 
                patience=12, visualization_interval=2):
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5, verbose=True)
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    train_losses = []
    val_losses = []
    
    model_size = get_model_size_mb(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️ Lightweight U-Net: {total_params:,} parâmetros")
    print(f"📏 Tamanho: {model_size:.2f} MB")
    
    if model_size <= 25:
        print(f"✅ Tamanho PERFEITO para Jetson Nano!")
    else:
        print(f"⚠️ Ainda pode ser grande para Jetson Nano")
    
    for epoch in range(num_epochs):
        # Treino
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f'Época {epoch+1:02d}/{num_epochs} [TREINO]')
        for images, masks in train_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
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
        
        print(f'📈 Época {epoch+1:02d}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}')
        
        if (epoch + 1) % visualization_interval == 0:
            print(f"🎨 Criando visualizações...")
            create_prediction_stack(model, val_loader, device, num_samples=5, epoch=epoch+1)
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping na época {epoch+1}")
            break
        
        scheduler.step(avg_val_loss)
    
    return train_losses, val_losses, early_stopping.best_loss

# ==================== SCRIPT PRINCIPAL ====================
def main():
    print("=" * 70)
    print("🏁 LIGHTWEIGHT U-NET PARA JETSON NANO")
    print("🎯 Target: 15-25 MB, 30+ FPS")
    print("=" * 70)
    
    # Configurações otimizadas
    IMAGES_PATH = "images"
    MASKS_PATH = "masks"
    IMG_SIZE = (256, 256)  # Manter 256x256
    BATCH_SIZE = 8  # Pode aumentar
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    PATIENCE = 4
    VISUALIZATION_INTERVAL = 2
    
    # Preparar dados
    train_dataset, val_dataset = prepare_data(IMAGES_PATH, MASKS_PATH, IMG_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"📚 Dataset treino: {len(train_dataset)} amostras")
    print(f"📚 Dataset validação: {len(val_dataset)} amostras")
    
    # Criar modelo leve
    model = LightweightUNet(in_channels=3, out_channels=1).to(device)
    
    model_size = get_model_size_mb(model)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"🏗️ Modelo Lightweight criado:")
    print(f"   📊 Parâmetros: {total_params:,}")
    print(f"   📏 Tamanho: {model_size:.2f} MB")
    print(f"   🧠 Features: Light Attention + Depthwise Separable Conv")
    
    # Treinar
    train_losses, val_losses, best_loss = train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        visualization_interval=VISUALIZATION_INTERVAL
    )
    
    # Resultados
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Treino', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validação', color='red', linewidth=2)
    plt.title('🏁 Lightweight U-Net - Loss', fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    models = ['Turbo\nU-Net', 'Lightweight\nU-Net', 'Target\nJetson']
    sizes = [142.93, model_size, 25]
    colors = ['red', 'green', 'blue']
    
    bars = plt.bar(models, sizes, color=colors, alpha=0.7)
    plt.title('Comparação de Tamanhos', fontweight='bold')
    plt.ylabel('Tamanho (MB)')
    plt.axhline(y=25, color='orange', linestyle='--', label='Limite Jetson')
    
    for bar, size in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{size:.1f} MB', ha='center', fontweight='bold')
    
    plt.subplot(1, 3, 3)
    info_text = f"""
    🏁 LIGHTWEIGHT U-NET
    
    📏 Tamanho: {model_size:.1f} MB
    📊 Parâmetros: {total_params:,}
 
    """
    plt.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('lightweight_unet_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Visualização final
    print("🎨 Criando visualização final...")
    create_prediction_stack(model, val_loader, device, num_samples=5, epoch=NUM_EPOCHS)
    
 
    torch.save(model.state_dict(), 'lightweight_unet_jetson.pth')
    
    # TorchScript
    model.eval()
    example_input = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save('lightweight_unet_traced.pt')
    
    print("=" * 70)
    print(f"✅ LIGHTWEIGHT U-NET CONCLUÍDA!")
    print(f"🏆 Melhor validation loss: {best_loss:.6f}")
    print(f"📏 Tamanho final: {model_size:.2f} MB")
    print(f"💾 Modelos guardados:")
    print(f"   🏁 lightweight_unet_jetson.pth")
    print(f"   🚀 lightweight_unet_traced.pt")
    print("=" * 70)

if __name__ == "__main__":
    main()

