import clip

CLIP_MODEL = "ViT-B/32"

if __name__ == "__main__":
    clip.load(CLIP_MODEL, device="cpu", download_root="./clip")