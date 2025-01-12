import torch
import torchvision.models as models
import torchvision.utils as vutils
from tqdm import trange

device = torch.device("mps")


def save_img(img: torch.Tensor, name: str):
    # Normalize and save
    img = (img - img.min()) / (img.max() - img.min())
    img = vutils.make_grid(img, nrow=int(img.shape[0] ** 0.5))
    vutils.save_image(img, name)


# Load pretrained model
model = models.mobilenet_v3_small().eval()
model.to(device)

# Initialize random image
img = torch.randn(64, 3, 224, 224, requires_grad=True, device=device)
input = img.clone()

save_img(img, "input.png")

# Use an optimizer
optimizer = torch.optim.Adam([img], lr=1e-1, maximize=True)

pbar = trange(2000)
for _ in pbar:
    optimizer.zero_grad()
    output = model(img)
    loss = output[:, 0].norm()
    loss.backward()
    optimizer.step()

    pbar.set_postfix_str(f"loss: {loss:.2e}")

save_img(img, "output.png")

diff = img - input
save_img(diff, "diff.png")
