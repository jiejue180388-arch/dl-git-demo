import os, random, torch, yaml
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import SmallCNN
from tqdm.auto import tqdm

seed=42
random.seed(seed)
torch.manual_seed(seed)

config = {"epochs":1,"batch_size":128,"lr":1e-3,"experiment":"baseline"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2)

model=SmallCNN().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=config["lr"])

exp_dir=os.path.join("experiments", config["experiment"])
os.makedirs(exp_dir, exist_ok=True)
with open(os.path.join(exp_dir, "config.yaml"), "w") as f: yaml.dump(config, f)

running_loss=0.0
model.train()
for epoch in range(config["epochs"]):
    loop=tqdm(trainloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    for xb,yb in loop:
        xb,yb=xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out=model(xb)
        loss=criterion(out,yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=running_loss/(loop.n+1))

torch.save(model.state_dict(), os.path.join(exp_dir, "model.pth"))
with open(os.path.join(exp_dir, "results.txt"), "w") as f:
    f.write(f"train_loss:{running_loss/len(trainloader):.4f}\n")
print("Done")
