import torch
import pygame
from medmnist import PathMNIST
import torchvision.transforms as transforms
from models import NCA, CNNBaseline
import os
import shutil

model_path = "models/best_nca_pathmnist.pth"
baseline_path = "models/best_cnn_pathmnist.pth"

cache_dir = os.path.expanduser("~/.medmnist")
os.makedirs(cache_dir, exist_ok=True)

dataset = "pathmnist"
resolutions = ["", "_64", "_128", "_224"]

for res in resolutions:
    filename = f"{dataset}{res}.npz"
    src = os.path.join("./data", filename)
    dst = os.path.join(cache_dir, filename)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print(f"Copied {filename} to cache.")
    else:
        print(f"File not found in ./data: {filename}")

pygame.init()

font = pygame.font.SysFont(None, 30)
big_font = pygame.font.SysFont(None, 40)

BG_COLOR = (30, 30, 30)
BTN_COLOR = (70, 70, 70)
BTN_HOVER = (100, 100, 100)
BTN_ACTIVE = (0, 150, 255)
TEXT_COLOR = (255, 255, 255)
THUMB_SIZE = 200
THUMB_MARGIN = 40

res_options = [("Original 28x28", ""), ("64x64", "_64"), ("128x128", "_128"), ("224x224", "_224")]
selected_res_idx = 0

transform = transforms.Compose([transforms.ToTensor()])

def load_all_thumbnails():
    print("Loading all datasets and thumbnails... This may take a moment.")
    all_thumbs = {}
    for label, suffix in res_options:
        size = int(suffix[1:]) if suffix else 28
        print(f"Loading dataset for resolution {label} (size={size})...")
        dataset = PathMNIST(split='test', size=size, transform=transform, download=False)
        thumbs = []
        for i in range(10):
            img, lbl = dataset[i]
            img = img.clamp(0, 1)
            img = (img * 255).byte().cpu().numpy()
            img = img.transpose(1, 2, 0)
            surf = pygame.surfarray.make_surface(img)
            surf = pygame.transform.scale(surf, (THUMB_SIZE, THUMB_SIZE))
            thumbs.append((surf, lbl))
        all_thumbs[suffix] = thumbs
    print("Finished loading all thumbnails.")
    return all_thumbs

def draw_button(screen, rect, text, hovered, active):
    color = BTN_COLOR
    if active:
        color = BTN_ACTIVE
    elif hovered:
        color = BTN_HOVER
    pygame.draw.rect(screen, color, rect)
    txt_surf = font.render(text, True, TEXT_COLOR)
    txt_rect = txt_surf.get_rect(center=rect.center)
    screen.blit(txt_surf, txt_rect)

def tensor_to_surface(tensor):
    tensor = tensor.clamp(0, 1)
    tensor = (tensor * 255).byte().cpu().numpy()
    img = tensor.transpose(1, 2, 0)
    return pygame.surfarray.make_surface(img)

def main():
    global selected_res_idx
    screen = pygame.display.set_mode((1200, 700))  # Moved inside main
    pygame.display.set_caption("Select Resolution and Image")

    all_thumbs = load_all_thumbnails()
    thumbs = all_thumbs[res_options[selected_res_idx][1]]
    selected_img_idx = None
    running = True
    stage = "select"

    clock = pygame.time.Clock()

    btn_width = 180
    btn_height = 40
    btn_spacing = 20
    btns = []
    for i, (text, _) in enumerate(res_options):
        rect = pygame.Rect(20 + i * (btn_width + btn_spacing), 20, btn_width, btn_height)
        btns.append(rect)

    step_idx = 0
    vis_clock = pygame.time.Clock()
    rgb_steps = None
    vis_screen = None
    true_label = None
    pred_label = None
    baseline_label = None

    while running:
        if stage == "select":
            screen.fill(BG_COLOR)
            mx, my = pygame.mouse.get_pos()
            clicked = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    clicked = True

            for i, rect in enumerate(btns):
                hovered = rect.collidepoint(mx, my)
                active = (i == selected_res_idx)
                draw_button(screen, rect, res_options[i][0], hovered, active)

                if hovered and clicked:
                    if selected_res_idx != i:
                        selected_res_idx = i
                        thumbs = all_thumbs[res_options[selected_res_idx][1]]
                        selected_img_idx = None

            cols = 5
            start_x = 20
            start_y = 100

            thumb_rects = []
            for idx, (surf, label) in enumerate(thumbs):
                col = idx % cols
                row = idx // cols
                x = start_x + col * (THUMB_SIZE + THUMB_MARGIN)
                y = start_y + row * (THUMB_SIZE + THUMB_MARGIN)
                rect = pygame.Rect(x, y, THUMB_SIZE, THUMB_SIZE)
                thumb_rects.append(rect)

                border_color = BTN_ACTIVE if idx == selected_img_idx else (255, 255, 255)
                pygame.draw.rect(screen, border_color, rect.inflate(4, 4), 3)
                screen.blit(surf, (x, y))

                label_surf = font.render(f"Label: {label}", True, TEXT_COLOR)
                label_rect = label_surf.get_rect(midtop=(x + THUMB_SIZE // 2, y + THUMB_SIZE + 5))
                screen.blit(label_surf, label_rect)

            if clicked:
                for idx, rect in enumerate(thumb_rects):
                    if rect.collidepoint(mx, my):
                        selected_img_idx = idx
                        break

            screen.blit(font.render("Click a resolution button.", True, TEXT_COLOR), (20, 620))
            screen.blit(font.render("Click an image thumbnail to select.", True, TEXT_COLOR), (20, 645))
            screen.blit(font.render("Press ENTER to start visualization when image is selected.", True, TEXT_COLOR), (20, 670))

            keys = pygame.key.get_pressed()
            if selected_img_idx is not None and keys[pygame.K_RETURN]:
                stage = "visualize"
                size = int(res_options[selected_res_idx][1][1:]) if res_options[selected_res_idx][1] else 28
                dataset = PathMNIST(split='test', size=size, transform=transform, download=False)
                x, label = dataset[selected_img_idx]
                x = x.unsqueeze(0)

                model = NCA()
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()

                baseline = CNNBaseline()
                baseline.load_state_dict(torch.load(baseline_path, map_location=torch.device('cpu')))
                baseline.eval()

                with torch.no_grad():
                    out, rgb_steps = model(x, visualize=True)
                    pred_label = out.argmax(dim=1).item()
                    true_label = label

                    out_baseline = baseline(x)
                    baseline_label = out_baseline.argmax(dim=1).item()


                vis_screen = pygame.display.set_mode((280, 280))
                pygame.display.set_caption(f"NCA PathMNIST Visualization (img {selected_img_idx}, res {size}x{size})")
                step_idx = 0

            pygame.display.flip()
            clock.tick(60)

        elif stage == "visualize":
            vis_screen.fill((0, 0, 0))
            if rgb_steps:
                img_tensor = rgb_steps[step_idx].squeeze(0)
                surf = tensor_to_surface(img_tensor)
                surf = pygame.transform.scale(surf, (280, 280))
                vis_screen.blit(surf, (0, 0))

                label_text = big_font.render(f"T: {true_label} P: {pred_label} B: {baseline_label}", True, TEXT_COLOR)
                vis_screen.blit(label_text, (10, 240))

                pygame.display.flip()
                step_idx = (step_idx + 1) % len(rgb_steps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        stage = "select"
                        vis_screen = None
                        screen = pygame.display.set_mode((1200, 700))
                        pygame.display.set_caption("Select Resolution and Image")

            vis_clock.tick(2)

    pygame.quit()

if __name__ == "__main__":
    main()

