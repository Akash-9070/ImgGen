import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer
from accelerate import Accelerator
import os
from PIL import Image
import json
from tqdm.auto import tqdm
import wandb
import argparse
from pathlib import Path

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer, transform=None, augment=True):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.base_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.augment = augment
        if augment:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ])

        with open(caption_file, 'r') as f:
            self.captions = json.load(f)

        self.image_files = []
        self.final_captions = []
        for img_file, caption in self.captions.items():
            full_path = self.image_dir / img_file
            if full_path.exists():
                self.image_files.append(full_path)
                self.final_captions.append(caption)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        
        if self.augment:
            image = self.augmentation(image)
        image = self.base_transform(image)

        caption = self.final_captions[idx]
        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'image': image,
            'input_ids': tokenized.input_ids[0],
            'attention_mask': tokenized.attention_mask[0]
        }

def prepare_training_data(image_folder):
    captions = {}
    image_folder = Path(image_folder)
    
    print("\nProvide detailed captions for each training image.")
    for img_file in image_folder.glob('*.[jp][pn][g]'):
        while True:
            caption = input(f"Caption for {img_file.name}: ").strip()
            if len(caption) >= 5 and not caption.startswith('.'):
                captions[img_file.name] = caption
                break
            print("Please provide a detailed caption (min 5 chars)")

    caption_file = image_folder / 'captions.json'
    with open(caption_file, 'w') as f:
        json.dump(captions, f, indent=4)

    return caption_file

def generate_samples(pipeline, prompts, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    images = pipeline(prompts, num_images_per_prompt=1).images
    for idx, image in enumerate(images):
        image.save(output_dir / f"sample_{idx}.png")
        wandb.log({f"sample_{idx}": wandb.Image(image)})

def fine_tune_model(
    pretrained_model_name="CompVis/stable-diffusion-v1-4",
    training_dir="training_images",
    output_dir="fine_tuned_model",
    num_epochs=100,
    learning_rate=1e-5,
    batch_size=1,
    validation_prompts=["test prompt 1", "test prompt 2"],
    save_steps=500,
    sample_steps=100,
):
    wandb.init(project="stable-diffusion-finetuning")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16"
    )

    if accelerator.is_local_main_process:
        wandb.config.update({
            "pretrained_model": pretrained_model_name,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs
        })

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name,
        subfolder="tokenizer"
    )

    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name,
        torch_dtype=torch.float32
    ).to(accelerator.device)

    if torch.cuda.device_count() > 1:
        pipeline.unet = DDP(pipeline.unet)

    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    caption_file = prepare_training_data(training_dir)
    dataset = CustomImageDataset(training_dir, caption_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        pipeline.unet.parameters(),
        lr=learning_rate
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs
    )

    pipeline.unet, optimizer, dataloader = accelerator.prepare(
        pipeline.unet, optimizer, dataloader
    )

    pipeline.unet.train()
    global_step = 0
    best_loss = float('inf')

    progress_bar = tqdm(range(num_epochs * len(dataloader)))
    progress_bar.set_description("Training Progress")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            with accelerator.accumulate(pipeline.unet):
                with autocast():
                    latents = pipeline.vae.encode(
                        batch["image"].to(dtype=torch.float32)
                    ).latent_dist.sample()
                    latents = latents * 0.18215

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, 
                        noise_scheduler.config.num_train_timesteps, 
                        (latents.shape[0],)
                    )
                    noisy_latents = noise_scheduler.add_noise(
                        latents, 
                        noise, 
                        timesteps
                    )

                    encoder_hidden_states = pipeline.text_encoder(
                        batch["input_ids"].to(accelerator.device)
                    )[0]

                    noise_pred = pipeline.unet(
                        noisy_latents, 
                        timesteps, 
                        encoder_hidden_states
                    ).sample

                    loss = torch.nn.functional.mse_loss(
                        noise_pred.float(), 
                        noise.float()
                    )

                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(
                    pipeline.unet.parameters(), 
                    1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1
            epoch_loss += loss.detach().item()

            if accelerator.is_local_main_process:
                wandb.log({
                    "loss": loss.item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0]
                })

                if global_step % sample_steps == 0:
                    pipeline.eval()
                    with torch.no_grad():
                        generate_samples(
                            pipeline,
                            validation_prompts,
                            output_dir / f"samples_{global_step}"
                        )
                    pipeline.train()

                if global_step % save_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    pipeline.save_pretrained(checkpoint_dir)
                    print(f"\nCheckpoint saved: {checkpoint_dir}")
                    print(f"Loss: {loss.item():.4f}")

            progress_bar.set_postfix(loss=loss.detach().item())

        epoch_loss = epoch_loss / len(dataloader)
        lr_scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            pipeline.save_pretrained(output_dir / "best_model")

    pipeline.save_pretrained(output_dir)
    print(f"\nTraining complete. Final model saved to {output_dir}")
    return pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    print("Starting fine-tuning process...")
    pipeline = fine_tune_model(**vars(args))

    prompt = input("\nEnter test prompt: ")
    image = pipeline(prompt).images[0]
    image.save(Path(args.output_dir) / "test_output.png")

if __name__ == "__main__":
    main()