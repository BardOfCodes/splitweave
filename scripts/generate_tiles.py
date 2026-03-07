"""
Generate tile images via LayerDiffuse + SDXL: sample prompts from word lists and
style vocabularies, run diffusion, filter by crop/size/color/occupancy, and write
images plus a metadata list. Config is loaded only from YAML/JSON via --config. Use --rank for parallel job spawning.

Prerequisites (install separately; not in splitweave core deps):
  - LayerDiffuse_DiffusersCLI repo: git clone https://github.com/lllyasviel/LayerDiffuse_DiffusersCLI
    Then install its requirements: cd LayerDiffuse_DiffusersCLI && pip install -r requirements.txt
  - Set ldcli_path in your config YAML to the cloned repo path.
"""

import argparse
import io
import json
import os
import random
import secrets
import string
import sys
import time
from pathlib import Path
from string import Template
from types import SimpleNamespace

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

import cv2
import numpy as np
import safetensors.torch as sf
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

# LayerDiffuse CLI path; edit for your environment. Used for parallel jobs via --rank.
LDCLI_PATH = "/users/aganesh8/data/aganesh8/projects/patterns/LayerDiffuse_DiffusersCLI"
sys.path.insert(0, LDCLI_PATH)
from diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from lib_layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder
from lib_layerdiffuse.utils import download_model


def load_config(path: Path) -> SimpleNamespace:
    """Load config from YAML or JSON. Returns a namespace (config.attr)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text()
    suf = path.suffix.lower()
    if suf in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise RuntimeError("YAML config requires PyYAML. Install with: pip install pyyaml")
        data = yaml.safe_load(text)
    elif suf == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension: {suf}. Use .yaml, .yml, or .json")
    if not data:
        raise ValueError("Config file is empty")
    return SimpleNamespace(**data)


# -----------------------------------------------------------------------------
# Prompt vocabulary: styles, color schemes, minimalism keywords, templates.
# -----------------------------------------------------------------------------
PROMPT_TEMPLATES = [
    Template("A minimal $style $second_term of a $noun $minimalism on a $color_scheme background."),
]

SECOND_TERMS = [
    "artwork", "design", "icon", "vector art", "illustration",
]

# Geometric and minimal, traditional, cultural, digital, decorative, etc.
STYLES = [
    "geometric", "symmetrical", "mandala-inspired", "flat design", "Scandinavian-inspired",
    "low-poly", "minimalist", "isometric", "line-art", "abstract", "origami-inspired",
    "pixel art", "stencil", "sketch", "doodle", "drawing", "watercolor", "oil painting",
    "charcoal", "ink and wash", "gouache", "pastel", "engraving", "woodcut", "monotype",
    "fresco", "art-deco", "art-nouveau", "cubism", "impressionism", "expressionism",
    "surrealism", "baroque", "rococo", "renaissance", "futurism", "bauhaus", "folk art",
    "japanese ukiyo-e", "chinese ink painting", "aztec pattern", "native american pattern",
    "african tribal art", "polynesian art", "aboriginal art", "icon", "illustration",
    "digital painting", "vector art", "vector cyberpunk", "vector steampunk",
    "vector futuristic", "vector synthwave", "vector vaporwave", "glitch art", "graffiti",
    "retro design", "pop art", "comic book style", "manga", "anime", "caricature", "chibi",
    "flat concept art", "flat game art", "superflat", "vector street art", "vector tattoo art",
    "flat floral", "damask", "flat arabesque", "victorian", "gothic", "neoclassical",
    "psychedelic", "mosaic", "kaleidoscopic", "tile art", "zentangle", "ornamental", "grunge",
    "vector blueprint", "vector diagram", "wireframe", "vector schematic", "vector cartography",
    "vector urban sketching", "vector architectural rendering",
]

MINIMALISM_KEYWORDS = [
    "with clean lines and simple shapes", "with solid colors and no gradients",
    "with a limited color palette", "with a monochromatic palette", "with minimal detailing",
    "with soft, rounded shapes", "with sharp, angular forms", "using only essential lines",
    "with minimalistic proportions", "with simple geometry", "with smooth curves",
    "with stylized proportions", "with subtle gradients", "with low visual complexity",
    "with balanced negative space", "with minimal textures", "with a simple two-tone color scheme",
    "with abstract and non-detailed features", "with a focus on form over texture",
    "with minimal use of shading", "with thin, uniform lines",
]

COLOR_SCHEMES = [
    "white with soft accent colors", "pastel tones with light neutrals",
    "light monochrome with a hint of color", "neutral tones with bold accents",
    "monochrome with a contrasting accent color", "earthy tones with muted accents",
    "muted shades with soft highlights", "cool greys and blues with soft whites",
    "warm tones like beige and peach with soft blues", "soft gradients with warm accents",
    "bold but minimal colors with neutral tones", "off-white with dark accents",
    "pale pinks and yellows with soft greys", "desaturated colors with sharp contrast",
    "soft gradients with pastel accents", "light gradients with warm hues",
    "subtle gradients with earthy undertones", "cool gradients with warm contrasts",
    "light greys with pastel gradients", "neutral gradients with bold, clean edges",
    "soft gradients with light neutral tones", "two-tone beige and off-white",
    "soft cream and pastel blues", "muted greens with dusty pink", "light grey with soft rose",
    "peach with soft lavender", "pastel greens with muted browns", "light sky blue with sandy beige",
    "soft coral with mint green", "pale yellow with cool greys", "light navy with soft turquoise",
    "dusty blues with light stone", "sage green with soft off-white", "pale aqua with soft peach",
    "warm neutrals with soft olive tones", "cool blues with light lavender",
    "warm beige with light sage green", "burnt orange with light greys", "warm greys with soft coral",
    "soft peach with cool mint", "warm gold with light cream", "taupe with soft teal",
    "sunset tones with soft greys", "desaturated blues with light warm neutrals",
    "off-white with pastel accents", "light grey with bold accents", "soft lavender with beige highlights",
    "light olive with warm peach", "muted navy with cream highlights", "cool mint with warm beige",
    "soft gold with pale peach", "pale blue with warm accents", "charcoal with light pastel highlights",
    "soft rose with light greys", "off-white with earthy highlights", "cool neutral tones with soft gold",
    "pastel purples with pale green", "light terracotta with soft teal", "cool grey with pale sage",
    "pale coral with soft jade", "warm beige with light lavender", "muted maroon with soft grey",
    "light plum with cool greys", "pale yellow with light terracotta", "light denim with pale gold",
    "dusty teal with light peach",
]


def generate_random_name(length: int = 8) -> str:
    """Return a random lowercase alphabetic string of given length."""
    return "".join(secrets.choice(string.ascii_lowercase) for _ in range(length))


def crop_image(image, padding: int = 0):
    """Crop to bounding box of non-zero alpha; return (PIL Image, success)."""
    img = np.asarray(image)
    gry = img[:, :, -1]
    coords = cv2.findNonZero(gry)
    if coords is None:
        return Image.fromarray(img), False
    x, y, w, h = cv2.boundingRect(coords)
    if w < 256 and h < 256:
        return Image.fromarray(img[y : y + h, x : x + w]), False
    cropped = img[y - padding : y + h + padding, x - padding : x + w + padding]
    return Image.fromarray(cropped), True


def get_size_after_compression(image, pil_input: bool = True) -> int:
    """Return file size in bytes after JPEG compression. Used to filter low-detail tiles."""
    if pil_input:
        pil_img = image
    else:
        pil_img = Image.fromarray((image * 255).astype(np.uint8))
    buf = io.BytesIO()
    rgb = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
    rgb.save(buf, format="JPEG", quality=85, optimize=True)
    return len(buf.getvalue())


def accept_tile(cropped_image, config: SimpleNamespace) -> bool:
    """Return True if the cropped image passes size, white/black, and occupancy checks."""
    size = get_size_after_compression(cropped_image, pil_input=True)
    weighted_lim = 5 * cropped_image.size[0] * cropped_image.size[1] / (512 * 512)
    if size <= weighted_lim:
        return False
    np_image = np.array(cropped_image)
    z = np_image[:, :, :3]
    w = (z > config.white_lim).mean()
    b = (z < config.black_lim).mean()
    if not (w < 0.90 and b < 0.90 and (w + b) < 0.90):
        return False
    last_channel = np_image[:, :, -1]
    if last_channel.max() == 0:
        return False
    avg_occ = last_channel.mean() / last_channel.max()
    if avg_occ > 0.8 or avg_occ < 0.2:
        return False
    return True


def load_pipeline(config: SimpleNamespace):
    """
    Load SDXL + LayerDiffuse models and return (pipeline, vae, transparent_decoder).
    Caller moves components to CUDA and creates the generator with the desired seed.
    """
    sdxl_name = config.sdxl_name
    tokenizer = CLIPTokenizer.from_pretrained(sdxl_name, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(sdxl_name, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(
        sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16"
    )
    text_encoder_2 = CLIPTextModel.from_pretrained(
        sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16"
    )
    vae = AutoencoderKL.from_pretrained(
        sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16"
    )
    unet = UNet2DConditionModel.from_pretrained(
        sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
    )
    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

    model_dir = Path(config.ldcli_path) / "models"
    path_attn = download_model(
        url="https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_attn.safetensors",
        local_path=str(model_dir / "ld_diffusers_sdxl_attn.safetensors"),
    )
    path_enc = download_model(
        url="https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_encoder.safetensors",
        local_path=str(model_dir / "ld_diffusers_sdxl_vae_transparent_encoder.safetensors"),
    )
    path_dec = download_model(
        url="https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_decoder.safetensors",
        local_path=str(model_dir / "ld_diffusers_sdxl_vae_transparent_decoder.safetensors"),
    )

    sd_offset = sf.load_file(path_attn)
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] if k in sd_offset else sd_origin[k] for k in sd_origin}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged

    transparent_decoder = TransparentVAEDecoder(path_dec)
    pipeline = KDiffusionStableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=None,
    )

    for obj in (text_encoder, text_encoder_2, unet, vae, transparent_decoder):
        obj.to("cuda")
    return pipeline, vae, transparent_decoder


def main():
    parser = argparse.ArgumentParser(description="Generate tiles; use --rank for parallel jobs.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML/JSON config (e.g. configs/tile_gen/default.yaml).",
    )
    parser.add_argument("--rank", type=int, default=10, help="Job rank (seed = seed_base + rank).")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.seed_base + args.rank
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random.seed(seed)

    nouns = Path(config.word_list_path).read_text().strip().split("\n")
    assert len(nouns) > 0, f"No nouns found in {config.word_list_path}"

    pipeline, vae, transparent_decoder = load_pipeline(config)
    
    Path(config.image_dir).mkdir(parents=True, exist_ok=True)
    all_names = []
    overall_count = 0
    default_negative = "face asymmetry, eyes asymmetry, deformed eyes, open mouth"

    while overall_count < config.n_tiles:
        init = overall_count
        st = time.time()
        prompt_list = []
        style = random.sample(STYLES, 1)[0]
        minimalism = random.sample(MINIMALISM_KEYWORDS, 1)[0]
        second_term = random.sample(SECOND_TERMS, 1)[0]
        color_scheme = random.sample(COLOR_SCHEMES, 1)[0]

        for i in range(config.n_prompts):
            if i % config.replace_prompt_every == 0:
                style = random.sample(STYLES, 1)[0]
                minimalism = random.sample(MINIMALISM_KEYWORDS, 1)[0]
                second_term = random.sample(SECOND_TERMS, 1)[0]
                color_scheme = random.sample(COLOR_SCHEMES, 1)[0]
            noun = random.sample(nouns, 1)[0]
            tmpl = random.sample(PROMPT_TEMPLATES, 1)[0]
            prompt = tmpl.substitute(
                noun=noun, style=style, second_term=second_term,
                minimalism=minimalism, color_scheme=color_scheme,
            ).replace("  ", " ")
            prompt_list.append(prompt)
            print(prompt)

        with torch.inference_mode():
            p_c, p_p, n_c, n_p = [], [], [], []
            for i in range(config.n_prompts):
                pos_c, pos_p = pipeline.encode_cropped_prompt_77tokens(prompt_list[i])
                neg_c, neg_p = pipeline.encode_cropped_prompt_77tokens(default_negative)
                p_c.append(pos_c)
                p_p.append(pos_p)
                n_c.append(neg_c)
                n_p.append(neg_p)
            positive_cond = torch.cat(p_c, 0)
            positive_pooler = torch.cat(p_p, 0)
            negative_cond = torch.cat(n_c, 0)
            negative_pooler = torch.cat(n_p, 0)
            unet = pipeline.unet
            initial_latent = torch.zeros(
                (config.n_prompts, 4, 144, 112), dtype=unet.dtype, device=unet.device
            )
            latents = pipeline(
                initial_latent=initial_latent,
                strength=1.0,
                num_inference_steps=config.n_diffusion_steps,
                batch_size=config.n_per_concept,
                prompt_embeds=positive_cond,
                negative_prompt_embeds=negative_cond,
                pooled_prompt_embeds=positive_pooler,
                negative_pooled_prompt_embeds=negative_pooler,
                generator=generator,
                guidance_scale=config.guidance_scale,
            ).images
            latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
            images = []
            for cur_latents in torch.chunk(latents, config.n_per_concept, 0):
                result_list, _ = transparent_decoder(vae, cur_latents)
                for image in result_list:
                    images.append(image)

        name = generate_random_name()
        for index, image in enumerate(images):
            cropped_image, crop_ok = crop_image(image, padding=0)
            if not crop_ok:
                print("Failed crop condition")
                continue
            if not accept_tile(cropped_image, config):
                continue
            all_names.append(name)
            out_path = os.path.join(config.image_dir, f"{name}_{index}.png")
            cropped_image.save(out_path)
            overall_count += 1
            print(name)

        et = time.time()
        print(f"Time taken for {overall_count - init} images: {et - st}")

    metadata_file = os.path.join(config.metadata_dir, f"tile_names_{args.rank}.txt")
    Path(metadata_file).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w") as f:
        for name in all_names:
            f.write(name + "\n")


if __name__ == "__main__":
    main()
