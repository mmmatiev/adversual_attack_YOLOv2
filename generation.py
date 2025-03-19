"""
generation.py

код для обучения adversarial patch.

Режимы запуска:
  python generation.py obj        # минимизация objectness (loss = obj_score)
  python generation.py cls        # минимизация class score (loss = cls_score)
  python generation.py obj-cls    # минимизация obj_score * cls_score

Также поддерживается возобновление обучения (resume_path) и инициализация патча из изображения (init_image_path).

Чекпоинты сохраняются в папку saved_patches/.
"""
import time
import gc
import torch
import torch.optim as optim
import os
import sys
import torch.nn.functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter
from PIL import Image

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Импорт модулей из проекта
from load_data import (
    Darknet,
    PatchApplier,
    PatchTransformer,
    MaxProbExtractor,
    NPSCalculator,
    TotalVariation,
    InriaDataset
)


class BaseConfig:
    """
    Базовая конфигурация для обучения.
    """
    def __init__(self):
        # Пути к данным и моделям
        self.img_dir = "inria/Train/pos"
        self.lab_dir = "inria/Train/pos/yolo-labels"
        self.cfgfile = "cfg/yolo.cfg"
        self.weightfile = "weights/yolov2.weights"
        self.printfile = "non_printability/30values.txt"

        # Параметры
        self.patch_size = 400
        self.start_learning_rate = 0.03
        self.batch_size = 5
        self.max_tv = 0.2

        # По умолчанию минимизируем obj_score
        self.loss_target = lambda obj, cls: obj


        self.scheduler_factory = lambda optimizer: ReduceLROnPlateau(
            optimizer, mode='min', patience=50
        )


def init_tensorboard(run_name=None):
    """
    Инициализация SummaryWriter для локальных логов.
    """
    if run_name:
        time_str = time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(f"runs/{time_str}_{run_name}")
    else:
        writer = SummaryWriter()
    return writer

def generate_patch(cfg: BaseConfig, patch_type="gray", image_path=None):
    """
    Генерирует начальный патч.
    """
    patch_size = cfg.patch_size
    if patch_type == "image" and image_path:
        img = Image.open(image_path).convert("RGB")
        tfm = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor()
        ])
        adv_patch = tfm(img)
        print(f"[INFO] Патч инициализирован изображением: {image_path}")
        return adv_patch
    elif patch_type == "gray":
        return torch.full((3, patch_size, patch_size), 0.5)
    elif patch_type == "random":
        return torch.rand((3, patch_size, patch_size))
    else:
        raise ValueError("patch_type должен быть 'gray', 'random' или 'image'.")

def save_checkpoint(epoch, adv_patch_cpu, optimizer, out_dir="saved_patches", interrupted=False):
    """
    Сохраняет чекпоинт и изображение патча.
    """
    os.makedirs(out_dir, exist_ok=True)  # Создаем папку, если её нет
    suffix = f"_epoch_{epoch}"
    if interrupted:
        suffix += "_INT"
    ckpt_path = f"{out_dir}/patch_checkpoint{suffix}.pt"
    torch.save({
        'epoch': epoch,
        'patch': adv_patch_cpu.detach().cpu(),
        'optimizer_state': optimizer.state_dict()
    }, ckpt_path)
    print(f"[INFO] Чекпоинт сохранён: {ckpt_path}")

    img_path = f"{out_dir}/patch_{suffix}.jpg"
    pil_img = transforms.ToPILImage()(adv_patch_cpu.detach().cpu())
    pil_img.save(img_path)
    print(f"[INFO] Патч сохранён как изображение: {img_path}")

def load_checkpoint(ckpt_path):
    """
    Загружает чекпоинт (.pt) и возвращает словарь.
    """
    print(f"[INFO] Загружаем чекпоинт: {ckpt_path}")
    return torch.load(ckpt_path)


def generation(mode_key, resume_path=None, init_image_path=None):
    """
    Основная функция обучения adversarial patch.
    """
    # 1) Создаём конфигурацию
    cfg = BaseConfig()

    # Настраиваем loss_target в зависимости от режима
    if mode_key == 'obj':
        cfg.loss_target = lambda obj_score, cls_score: obj_score
    elif mode_key == 'cls':
        cfg.loss_target = lambda obj_score, cls_score: cls_score
    elif mode_key == 'obj-cls':
        cfg.loss_target = lambda obj_score, cls_score: obj_score * cls_score
    else:
        raise ValueError(f"Неверный режим: {mode_key}. Ожидается: obj, cls или obj-cls.")

    # Определяем папку для сохранения чекпоинтов
    mode_to_dir = {
        'obj': 'saved_patches_obj',
        'cls': 'saved_patches_cls',
        'obj-cls': 'saved_patches_cls-obj'
    }
    out_dir = mode_to_dir[mode_key]
    os.makedirs(out_dir, exist_ok=True)  # Создаем папку, если её нет
    # 2) Инициализируем модель YOLO
    darknet_model = Darknet(cfg.cfgfile)
    darknet_model.load_weights(cfg.weightfile)
    darknet_model.eval().cuda()

    # 3) Вспомогательные модули
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()
    prob_extractor = MaxProbExtractor(0, 80, cfg).cuda()
    nps_calculator = NPSCalculator(cfg.printfile, cfg.patch_size).cuda()
    total_variation = TotalVariation().cuda()

    # 4) Инициализируем локальное логирование TensorBoard
    writer = init_tensorboard(run_name=mode_key)

    # 5) Генерируем или загружаем патч
    adv_patch_cpu = None
    start_epoch = 0
    checkpoint = None
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path)
        start_epoch = checkpoint['epoch'] + 1
        adv_patch_cpu = checkpoint['patch']
        adv_patch_cpu.requires_grad_(True)
    else:
        patch_type = 'image' if init_image_path else 'gray'
        adv_patch_cpu = generate_patch(cfg, patch_type, init_image_path)
        adv_patch_cpu.requires_grad_(True)

    # 6) Создаём оптимизатор
    optimizer = optim.Adam([adv_patch_cpu], lr=cfg.start_learning_rate, amsgrad=True)
    if checkpoint is not None and 'optimizer_state' in checkpoint:
        print("[INFO] Восстанавливаем состояние оптимизатора из чекпоинта.")
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    # 7) Планировщик
    scheduler = cfg.scheduler_factory(optimizer)

    # 8) DataLoader
    img_size = darknet_model.height
    train_dataset = InriaDataset(cfg.img_dir, cfg.lab_dir, max_lab=14, imgsize=img_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8
    )
    epoch_length = len(train_loader)
    print(f"[INFO] Длина одной эпохи: {epoch_length} итераций")

    # 9) Основной цикл обучения
    n_epochs = 1000
    et0 = time.time()
    try:
        for epoch in range(start_epoch, n_epochs):
            print(f"=== Начинается эпоха {epoch} ({mode_key}) ===")
            ep_det_loss = 0.0
            ep_nps_loss = 0.0
            ep_tv_loss = 0.0

            for i_batch, (img_batch, lab_batch) in enumerate(train_loader):
                img_batch = img_batch.cuda()
                lab_batch = lab_batch.cuda()
                adv_patch = adv_patch_cpu.cuda()

                # 1) преобразование патча (вращение, шум и т.д.)
                adv_batch_t = patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)

                # 2) наложение патча на изображения
                p_img_batch = patch_applier(img_batch, adv_batch_t)
                p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))

                # 3) прогон через YOLO
                output = darknet_model(p_img_batch)
                max_prob = prob_extractor(output)

                # 4) вычисление лоссов
                det_loss = torch.mean(max_prob)
                nps = nps_calculator(adv_patch)
                tv = total_variation(adv_patch)

                nps_loss = nps * 0.01
                tv_loss = tv * 2
                loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                ep_det_loss += det_loss.detach().cpu().item()
                ep_nps_loss += nps_loss.detach().cpu().item()
                ep_tv_loss += tv_loss.detach().cpu().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                adv_patch_cpu.data.clamp_(0, 1)

                if (i_batch % 5) == 0:
                    iteration = epoch * epoch_length + i_batch
                    writer.add_scalar("total_loss", loss.item(), iteration)
                    writer.add_scalar("loss/det_loss", det_loss.item(), iteration)
                    writer.add_scalar("loss/nps_loss", nps_loss.item(), iteration)
                    writer.add_scalar("loss/tv_loss", tv_loss.item(), iteration)
                    writer.add_scalar("misc/epoch", epoch, iteration)
                    writer.add_scalar("misc/lr", optimizer.param_groups[0]["lr"], iteration)
                    writer.add_image("patch", adv_patch_cpu, iteration)

                del p_img_batch, output, max_prob, det_loss, nps_loss, tv_loss, loss
                gc.collect()
                torch.cuda.empty_cache()

            num_batches = len(train_loader)
            ep_det_loss /= num_batches
            ep_nps_loss /= num_batches
            ep_tv_loss /= num_batches

            print(f"[Epoch {epoch}] det_loss={ep_det_loss:.4f}, nps_loss={ep_nps_loss:.4f}, tv_loss={ep_tv_loss:.4f}")
            scheduler.step(ep_det_loss)
            et1 = time.time()
            print(f"Эпоха {epoch} завершена за {et1 - et0:.2f} с")
            et0 = time.time()

            if epoch % 5 == 0:
                save_checkpoint(epoch, adv_patch_cpu, optimizer)

    except KeyboardInterrupt:
        print("[INFO] Обучение прервано (Ctrl+C). Сохраняем чекпоинт...")
        save_checkpoint(epoch, adv_patch_cpu, optimizer, interrupted=True)
        print("[INFO] Чекпоинт сохранён. Завершаем.")
        return

    print("[INFO] Обучение завершено! Сохраняем финальный патч...")
    save_checkpoint(n_epochs, adv_patch_cpu, optimizer)
    print("[INFO] Готово.")


def main():
    """
    Точка входа.
    Использование: python generation.py <obj|cls|obj-cls> [resume_path] [init_image_path]
    """
    if len(sys.argv) < 2:
        print("Usage: python generation.py <obj|cls|obj-cls> [resume_path] [init_image_path]")
        sys.exit()

    mode_key = sys.argv[1]  # 'obj', 'cls' или 'obj-cls'
    resume_path = sys.argv[2] if len(sys.argv) >= 3 else None
    init_image_path = sys.argv[3] if len(sys.argv) >= 4 else None

    generation(mode_key, resume_path=resume_path, init_image_path=init_image_path)


if __name__ == "__main__":
    main()



