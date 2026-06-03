#!/usr/bin/env python3
"""
CENG467 UNMT — Interactive Demo
===============================
A Gradio-based web interface to test the UNMT model live.
Perfect for the final project presentation or Colab demonstration.

Usage:
  python src/demo.py --checkpoint checkpoints/checkpoint_latest.pt
"""

import os
import sys
import argparse
import gradio as gr
import sentencepiece as spm
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils import load_config, get_device, setup_logging, add_base_args, resolve_path
from src.model import build_model
from src.generate import translate_lines

log = setup_logging("demo")

# Global variables to hold the loaded model and SP
MODEL = None
SP = None
DEVICE = None
CFG = None

def translate(text: str, direction: str, strategy: str, beam_size: int) -> str:
    """Translation callback for Gradio."""
    if not text.strip():
        return ""
        
    src_lang = "tr" if direction == "Turkish to English" else "en"
    strat = "greedy" if strategy == "Greedy (Fast)" else "beam"
    b_size = int(beam_size)
    
    try:
        results = translate_lines(
            [text], MODEL, SP, src_lang, DEVICE, 
            strategy=strat, beam_size=b_size, batch_size=1
        )
        return results[0]
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="UNMT Live Demo")
    add_base_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--share", action="store_true", default=True, help="Create a public Gradio link")
    args = parser.parse_args()

    global MODEL, SP, DEVICE, CFG
    CFG = load_config(args.config, args.base_dir)
    DEVICE = get_device()
    
    log.info("Loading SentencePiece...")
    proc_dir = resolve_path(CFG, "data", "processed_subdir")
    sp_path = os.path.join(proc_dir, f"{CFG['vocab']['model_prefix']}.model")
    SP = spm.SentencePieceProcessor(model_file=sp_path)
    
    log.info(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    MODEL = build_model(CFG, vocab_size=SP.get_piece_size(), pad_id=SP.pad_id())
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.to(DEVICE)
    MODEL.eval()
    
    log.info("Starting Gradio interface...")
    
    with gr.Blocks(title="UNMT Turkish-English") as demo:
        gr.Markdown("# 🇹🇷 🇬🇧 Unsupervised Neural Machine Translation")
        gr.Markdown(
            "Welcome to the interactive demo for Group 9's CENG467 Term Project. "
            "This model was trained entirely on **monolingual** Turkish and English text "
            "using Denoising Autoencoder and Iterative Back-Translation, without any parallel dictionaries or sentence pairs."
        )
        
        with gr.Row():
            with gr.Column():
                direction = gr.Radio(
                    ["Turkish to English", "English to Turkish"], 
                    value="Turkish to English", 
                    label="Translation Direction"
                )
                input_text = gr.Textbox(
                    lines=5, 
                    placeholder="Enter text to translate...", 
                    label="Source Text"
                )
                with gr.Row():
                    strategy = gr.Dropdown(
                        ["Greedy (Fast)", "Beam Search (Quality)"], 
                        value="Beam Search (Quality)", 
                        label="Decoding Strategy"
                    )
                    beam_size = gr.Slider(
                        minimum=1, maximum=8, value=4, step=1, 
                        label="Beam Size", interactive=True
                    )
                translate_btn = gr.Button("Translate", variant="primary")
                
            with gr.Column():
                output_text = gr.Textbox(
                    lines=8, 
                    label="Translation", 
                    interactive=False
                )
                
        translate_btn.click(
            fn=translate,
            inputs=[input_text, direction, strategy, beam_size],
            outputs=output_text
        )
        
        gr.Examples(
            examples=[
                ["Ankara, Türkiye'nin başkenti ve en kalabalık ikinci şehridir.", "Turkish to English", "Beam Search (Quality)", 4],
                ["Doğal dil işleme, bilgisayarların insan dilini anlamasını sağlayan bir yapay zeka alt dalıdır.", "Turkish to English", "Beam Search (Quality)", 4],
                ["Machine learning is a field of inquiry devoted to understanding and building methods that learn.", "English to Turkish", "Beam Search (Quality)", 4],
                ["The weather is extremely cold today, so you should wear a thick coat.", "English to Turkish", "Beam Search (Quality)", 4]
            ],
            inputs=[input_text, direction, strategy, beam_size],
        )

    demo.launch(share=args.share)

if __name__ == "__main__":
    main()
