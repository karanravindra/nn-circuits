import torch
import torchvision.models as models
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from tqdm import trange

device = torch.device("mps")  # change if needed, e.g. torch.device("cuda") or "cpu"


def save_img(img: torch.Tensor, name: str):
    # Normalize and save
    img = (img - img.min()) / (img.max() - img.min())
    img = vutils.make_grid(img, nrow=int(img.shape[0] ** 0.5))
    vutils.save_image(img, name)


# 1. Load a pretrained model
model = models.mobilenet_v3_small(pretrained=True).eval()
model.to(device)

# 2. Load an existing image
path_to_image = "n01440764_tench.JPEG"  # Update this to the actual path
img_pil = Image.open(path_to_image).convert("RGB")

# 3. Preprocess (resize to 224x224, convert to tensor)
transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
img_tensor = transform(img_pil).unsqueeze(0).to(device)
img_tensor.requires_grad_(True)

# Keep an original copy for comparison
input_tensor = img_tensor.clone().detach()

save_img(img_tensor, "input.png")

# 4. Use an optimizer; note 'maximize=True' in PyTorch 2.x
optimizer = torch.optim.AdamW([img_tensor], lr=1e-2, weight_decay=1e-2, maximize=True)

# 5. Gradient ascent loop
pbar = trange(200)
for _ in pbar:
    optimizer.zero_grad()
    output = model(img_tensor)
    # Class 0 activation
    loss = output[:, 0].norm()
    loss.backward()
    optimizer.step()

    pbar.set_postfix_str(f"loss: {loss:.2e}")

# 6. Save the final image
save_img(img_tensor, "output.png")

# 7. Save the difference image
diff = img_tensor - input_tensor
save_img(diff, "diff.png")
