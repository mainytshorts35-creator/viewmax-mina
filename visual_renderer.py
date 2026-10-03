import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Dict

class VisualRenderer:
    @staticmethod
    def reframe_to_916(frame_array: np.ndarray, target_w: int = 1080, target_h: int = 1920) -> np.ndarray:
        img = Image.fromarray(frame_array).convert("RGB")
        src_w, src_h = img.size
        
        scale = min(target_w / src_w, target_h / src_h)
        scaled_fg = img.resize((max(1, int(src_w * scale)), max(1, int(src_h * scale))), Image.Resampling.LANCZOS)
        
        bg_scale = max(target_w / src_w, target_h / src_h)
        background = img.resize((max(1, int(src_w * bg_scale)), max(1, int(src_h * bg_scale))), Image.Resampling.LANCZOS)
        background = background.crop((
            (background.width - target_w) // 2, (background.height - target_h) // 2, 
            (background.width + target_w) // 2, (background.height + target_h) // 2
        ))
        background = background.filter(ImageFilter.GaussianBlur(40))
        
        background.paste(scaled_fg, ((target_w - scaled_fg.width) // 2, (target_h - scaled_fg.height) // 2))
        return np.array(background)

    @staticmethod
    def draw_captions(frame_array: np.ndarray, t: float, words: List[Dict], style: str = "Hormozi") -> np.ndarray:
        if not words:
            return frame_array
            
        img = Image.fromarray(frame_array).convert("RGBA")
        canvas = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(canvas)
        w, h = img.size

        active_idx = -1
        for i, item in enumerate(words):
            if item["start"] <= t <= item["end"] + 0.15:
                active_idx = i
                break
                
        if active_idx == -1:
            return frame_array

        chunk_start = (active_idx // 3) * 3
        chunk_words = words[chunk_start : chunk_start + 3]
        
        try:
            font = ImageFont.truetype("Impact.ttf", int(h * 0.055))
        except:
            font = ImageFont.load_default()

        total_w = sum(draw.textbbox((0,0), m["word"], font=font)[2] for m in chunk_words) + (25 * len(chunk_words))
        curr_x = (w - total_w) // 2
        start_y = int(h * 0.75)

        for i, item in enumerate(chunk_words):
            is_active = (chunk_start + i == active_idx)
            
            if "Cyberpunk" in style:
                color = (0, 242, 254) if is_active else (255, 255, 255)
            else: # Default Hormozi
                color = (255, 215, 0) if is_active else (255, 255, 255)
                
            y_offset = -12 if is_active else 0
            
            for dx in [-4, 0, 4]:
                for dy in [-4, 0, 4]:
                    draw.text((curr_x + dx, start_y + y_offset + dy), item["word"], font=font, fill=(0,0,0, 255))
            
            draw.text((curr_x, start_y + y_offset), item["word"], font=font, fill=color)
            curr_x += draw.textbbox((0,0), item["word"], font=font)[2] + 25

        return np.array(Image.alpha_composite(img, canvas).convert("RGB"))
