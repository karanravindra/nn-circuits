import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.utils as vutils
import torchvision.transforms.functional as T
from tqdm import trange
from torchinfo import summary

DEVICE = torch.device("mps")
LAYER = 0
CHANNEL = 0

def save_img(img: torch.Tensor, name: str):
    img = T.resize(img, (64, 64), interpolation=T.InterpolationMode.NEAREST).cpu()
    img = vutils.make_grid(img, nrow=max(int(img.shape[0] ** 0.5), 4), normalize=True)
    vutils.save_image(img, f"./imgs/{name}_layer{LAYER}_channel{CHANNEL}.png")


model = models.vgg11()
model = model.features
model = nn.Sequential(*list(model.children())[:LAYER + 1])

summary(model, (1, 3, 128, 128), depth=2)

model.to(DEVICE)

print(model)

# Initialize random image
img = torch.rand(16, 3, 128, 128, requires_grad=True, device=DEVICE)
input = img.clone()

save_img(img, "input")

# Use an optimizer
optimizer = torch.optim.AdamW([img], lr=1e-1, maximize=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

pbar = trange(5000)
for _ in pbar:
    optimizer.zero_grad()
    output = model(img)[:, CHANNEL, :, :]
    loss = output.norm()
    loss.backward()
    optimizer.step()

    pbar.set_postfix_str(f"loss: {loss:.2e}")
    scheduler.step()

save_img(img, "output")
# save_img(model[LAYER].weight, "layer")
# save_img(model[LAYER].weight[CHANNEL], "weight")
