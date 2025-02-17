import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from liveplot import LivePlot

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 전처리
def get_smile(attr):
    return attr[31]  # 웃음 레이블 추출

transform_train = transforms.Compose([
    transforms.RandomCrop([178, 178]),
    transforms.RandomHorizontalFlip(),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.CenterCrop([178, 178]),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])

# 데이터셋 로드
image_path = './'

celeba_train_dataset = torchvision.datasets.CelebA(
    image_path, split='train',
    target_type='attr', download=False,
    transform=transform_train, target_transform=get_smile
)

celeba_valid_dataset = torchvision.datasets.CelebA(
    image_path, split='valid',
    target_type='attr', download=False,
    transform=transform, target_transform=get_smile
)

celeba_test_dataset = torchvision.datasets.CelebA(
    image_path, split='test',
    target_type='attr', download=False,
    transform=transform, target_transform=get_smile
)

# 데이터셋 서브셋 가져오기
celeba_train_dataset = Subset(celeba_train_dataset, torch.arange(16000))
celeba_valid_dataset = Subset(celeba_valid_dataset, torch.arange(1000))

# 데이터로더 작성
batch_size = 32
train_dl = DataLoader(celeba_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(celeba_valid_dataset, batch_size, shuffle=False)
test_dl = DataLoader(celeba_test_dataset, batch_size, shuffle=False)


# CNN 모델 정의
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.5),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.5),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=8),
    nn.Flatten(),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
model.to(device)  # 모델을 GPU로 이동


def train(model, num_epochs, train_dl, valid_dl):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    plotter = LivePlot()  # LivePlot 객체 생성
    
    for epoch in range(num_epochs):
        model.train()
        loss_tmp = 0
        accuracy_tmp = 0
        
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch.float())
            loss.backward()
            optimizer.step()
            
            loss_tmp += loss.item() * y_batch.size(0)
            is_correct = ((pred >= 0.5).float() == y_batch).float().cpu()
            accuracy_tmp += is_correct.sum()
        
        loss_train = loss_tmp / len(train_dl.dataset)
        acc_train = accuracy_tmp / len(train_dl.dataset)
        
        # 검증 루프
        model.eval()
        with torch.no_grad():
            loss_tmp = 0
            accuracy_tmp = 0
            
            for x_batch, y_batch in valid_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch.float())
                loss_tmp += loss.item() * y_batch.size(0)
                is_correct = ((pred >= 0.5).float() == y_batch).float().cpu()
                accuracy_tmp += is_correct.sum()
        
        loss_valid = loss_tmp / len(valid_dl.dataset)
        acc_valid = accuracy_tmp / len(valid_dl.dataset)
        
        plotter.update_accuracy_train(epoch, acc_train)  # 그래프 업데이트
        print(f'에포크 {epoch+1} 정확도: {acc_train:.4f} 검증 정확도: {acc_valid:.4f}')
    
    plotter.show()  # 최종 그래프 출력
    
    return loss_train, loss_valid, acc_train, acc_valid



torch.manual_seed(1)
num_epochs = 100
hist = train(model, num_epochs, train_dl, valid_dl)