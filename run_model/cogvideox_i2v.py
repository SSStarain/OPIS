#!/usr/bin/env python3
"""Minimal offline CogVideoX-5B-I2V inference entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image, ImageOps


NATIVE_SIZE = (720, 480)


def prepare_native_image(image: Image.Image, output_size: tuple[int, int]) -> Image.Image:
    """Letterbox the requested canvas for fixed-resolution CogVideoX inference."""
    output_width, output_height = output_size
    native_width, native_height = NATIVE_SIZE
    image = ImageOps.fit(image.convert("RGB"), output_size, method=Image.Resampling.LANCZOS)
    content_height = round(native_width * output_height / output_width)
    content_height -= content_height % 2
    content = image.resize((native_width, content_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", NATIVE_SIZE, "black")
    canvas.paste(content, (0, (native_height - content_height) // 2))
    return canvas


def restore_output_frame(frame: Image.Image, output_size: tuple[int, int]) -> Image.Image:
    """Remove the inference letterbox and restore the requested benchmark canvas."""
    native_width, native_height = NATIVE_SIZE
    output_width, output_height = output_size
    content_height = round(native_width * output_height / output_width)
    content_height -= content_height % 2
    top = (native_height - content_height) // 2
    frame = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    return frame.crop((0, top, native_width, top + content_height)).resize(
        output_size, Image.Resampling.LANCZOS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--num-frames", required=True, type=int)
    parser.add_argument("--fps", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", default=50, type=int)
    args = parser.parse_args()

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    output_size = (args.width, args.height)
    image = prepare_native_image(load_image(args.image), output_size)
    frames = pipe(
        height=NATIVE_SIZE[1], width=NATIVE_SIZE[0], prompt=args.prompt,
        image=image, num_inference_steps=args.steps,
        num_frames=args.num_frames, use_dynamic_cfg=True, guidance_scale=6.0,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).frames[0]
    frames = [restore_output_frame(frame, output_size) for frame in frames]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output), fps=args.fps)


if __name__ == "__main__":
    main()
