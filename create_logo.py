#!/usr/bin/env python3
"""
Script to create a simple logo for the RAG PDF application
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_simple_logo():
    """Create a simple logo image"""
    
    # Create image
    width, height = 200, 80
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Draw text
    text = "🤖 RAG PDF"
    
    if font:
        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width, text_height = 100, 20
    
    # Center the text
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw background rectangle
    draw.rectangle([10, 10, width-10, height-10], fill='#f0f2f6', outline='#1f77b4', width=2)
    
    # Draw text
    draw.text((x, y), text, fill='#1f77b4', font=font)
    
    # Save image
    logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
    image.save(logo_path)
    print(f"Logo created: {logo_path}")
    
    return logo_path

if __name__ == "__main__":
    try:
        create_simple_logo()
        print("Logo created successfully!")
    except Exception as e:
        print(f"Error creating logo: {str(e)}")
        print("You can run the app without logo - it's already fixed in main.py")