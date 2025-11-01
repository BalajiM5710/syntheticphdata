import os
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import AdamW
from PIL import Image
from datasets import Dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A
import numpy as np
import requests

# ===================================================================
# 1. CONFIGURATION (Tuned for M2 Max)
# ===================================================================
class TrainConfig:
    # --- Paths ---
    ROOT_DIR = "/Users/5078091/syntheticphdata/donut_FT_weights_M2_v1" # Use a new folder for this run
    DATA_DIR = "/Users/5078091/syntheticphdata/final_dataset_syn"
    
    # --- Model ---
    BASE_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"
    
    # --- Training Hyperparameters (M2 MAX OPTIMIZATIONS) ---
    # With 64GB unified memory, we can use a larger batch size directly.
    BATCH_SIZE = 8
    # We no longer need gradient accumulation with a larger batch size.
    ACCUMULATE_GRAD_BATCHES = 1
    MAX_EPOCHS = 50  # EarlyStopping will find the true best epoch.
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.05
    
    # --- Hardware Configuration (M2 MAX OPTIMIZATIONS) ---
    ACCELERATOR = "mps"
    # `mps` backend requires 32-bit precision for stability.
    PRECISION = 32
    # Your M2 Max can handle more parallel data loading workers.
    NUM_WORKERS = 4

# Create an instance of the configuration
config = TrainConfig()

# Create output directories
os.makedirs(config.ROOT_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(config.ROOT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===================================================================
# 2. DATA AUGMENTATION (No changes)
# ===================================================================
augmentation_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.Sharpen(p=0.3),
    A.ImageCompression(quality_lower=85, quality_upper=95, p=0.4),
], bbox_params=None)

# ===================================================================
# 3. DATA MODULE (No changes to logic, just uses config)
# ===================================================================
class DonutDataModule(pl.LightningDataModule):
    def __init__(self, config, processor):
        super().__init__()
        self.config = config
        self.processor = processor
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # This function correctly creates the 90/10 split. No changes needed.
        records = []
        img_dir = os.path.join(self.config.DATA_DIR, "images")
        ann_dir = os.path.join(self.config.DATA_DIR, "annotations")
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith((".jpg", ".png")):
                continue
            name = img_name.rsplit(".", 1)[0]
            records.append({
                "image_path": os.path.join(img_dir, img_name),
                "label": self._make_label(os.path.join(ann_dir, f"{name}.json"))
            })
        
        full_dataset = Dataset.from_list(records)
        split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        self.train_dataset = split_dataset["train"]
        self.val_dataset = split_dataset["test"]
        print(f"✓ Data setup complete. Train samples: {len(self.train_dataset)}, Validation samples: {len(self.val_dataset)}")

    # Inside the DonutDataModule class in your training script
    def _make_label(self, json_path):
        with open(json_path, 'r') as f: data = json.load(f)
        header = data["gt_parse"]["header"]
        items = data["gt_parse"]["items"]
        summary = data["gt_parse"]["summary"]
        
        label = "<s_invoice>"
        
        # Explicitly handle all known header fields
        for k in ["invoice_number", "invoice_date", "seller", "client", "shipping_address", "seller_tax_id", "client_tax_id", "client_phone"]:
            if k in header and header[k]: # Check if key exists and is not empty
                v = header[k]
                label += f"<{k}>{v}</{k}>"
        
        label += "<items>"
        for item in items:
            # Loop through all item keys, this is fine
            for k, v in item.items():
                label += f"<{k}>{v}</{k}>"
            label += "<sep/>"
        label += "</items>"
        
        label += "<summary>"
        # Loop through all summary keys, this is fine
        for k, v in summary.items():
            label += f"<{k}>{v}</{k}>"
        label += "</summary></s_invoice>"
        return label

    def _collate_fn(self, batch, is_train=False):
        images_pil = [Image.open(x["image_path"]).convert("RGB") for x in batch]
        texts = [x["label"] for x in batch]
        
        if is_train:
            images_np = [np.array(img) for img in images_pil]
            augmented_images = [augmentation_pipeline(image=img)['image'] for img in images_np]
            pixel_values = self.processor(augmented_images, return_tensors="pt").pixel_values
        else:
            pixel_values = self.processor(images_pil, return_tensors="pt").pixel_values

        labels = self.processor.tokenizer(
            texts, add_special_tokens=False, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values, "labels": labels}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True,
            collate_fn=lambda batch: self._collate_fn(batch, is_train=True),
            num_workers=self.config.NUM_WORKERS
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.config.BATCH_SIZE,
            collate_fn=lambda batch: self._collate_fn(batch, is_train=False),
            num_workers=self.config.NUM_WORKERS
        )

# ===================================================================
# 4. LIGHTNING MODULE (No changes to logic)
# ===================================================================
class DonutPLModule(pl.LightningModule):
    def __init__(self, config, model, processor):
        super().__init__()
        self.config = config
        self.model = model
        self.processor = processor
        self.save_hyperparameters(ignore=['model', 'processor'])

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = AdamW(trainable_params, lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        return optimizer

# ===================================================================
# 5. MAIN TRAINING FUNCTION
# ===================================================================
def main(config):
    # --- Setup Model and Processor ---
    processor = DonutProcessor.from_pretrained(config.BASE_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(config.BASE_MODEL)
    model.gradient_checkpointing_enable()

    new_tokens = ["<s_invoice>", "</s_invoice>", "<items>", "</items>", "<summary>", "</summary>", "<sep/>"]
    processor.tokenizer.add_tokens(new_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_invoice>"])[0]
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    for param in model.encoder.parameters():
        param.requires_grad = False

    # --- Setup Data ---
    data_module = DonutDataModule(config, processor)

    # --- Setup Callbacks and Logger ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="donut-best",
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode="min"
    )

    logger = TensorBoardLogger(config.ROOT_DIR, name="logs")
    
    # --- Initialize Trainer ---
    pl_module = DonutPLModule(config, model, processor)
    
    trainer = pl.Trainer(
        default_root_dir=config.ROOT_DIR,
        accelerator=config.ACCELERATOR,
        devices=1,
        max_epochs=config.MAX_EPOCHS,
        precision=config.PRECISION,
        gradient_clip_val=1.0,
        val_check_interval=0.5,
        log_every_n_steps=10,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accumulate_grad_batches=config.ACCUMULATE_GRAD_BATCHES,
        num_sanity_val_steps=0
    )

    # --- Run Training ---
    print("Starting training with M2 Max optimizations...")
    trainer.fit(pl_module, datamodule=data_module)
    print("✓ Training finished or stopped early.")

    # --- Save Final Production-Ready Model ---
    final_save_path = os.path.join(config.ROOT_DIR, "donut_final_model")
    
    if not checkpoint_callback.best_model_path:
        print("No best model found. Saving last model instead.")
        best_path = checkpoint_callback.last_model_path
    else:
        best_path = checkpoint_callback.best_model_path
        
    print(f"Loading best model from: {best_path}")
    best_pl_module = DonutPLModule.load_from_checkpoint(best_path)
    best_model = best_pl_module.model

    for param in best_model.encoder.parameters():
        param.requires_grad = True

    print(f"Saving best model and processor to: {final_save_path}")
    best_model.save_pretrained(final_save_path)
    processor.save_pretrained(final_save_path)
    
    print("✓ All tasks complete!")

if __name__ == '__main__':
    main(config)