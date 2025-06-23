import json
import torch
import numpy as np
from huggingface_hub import snapshot_download

from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
from transformers import AutoTokenizer, T5ForConditionalGeneration
from modelling_deberta_v2 import DebertaV2ForTokenClassificationRegression

from diffusers import DDPMScheduler
from models import MusicAudioDiffusion


class MusicFeaturePredictor:
    def __init__(self, path, device="cuda:0", cache_dir=None, local_files_only=False):
        self.beats_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-large",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.beats_model = DebertaV2ForTokenClassificationRegression.from_pretrained(
            "microsoft/deberta-v3-large",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.beats_model.eval()
        self.beats_model.to(device)

        beats_ckpt = f"{path}/beats/microsoft-deberta-v3-large.pt"
        beats_weight = torch.load(beats_ckpt, map_location="cpu")
        self.beats_model.load_state_dict(beats_weight)

        self.chords_tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-large",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.chords_model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-large",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.chords_model.eval()
        self.chords_model.to(device)

        chords_ckpt = f"{path}/chords/flan-t5-large.bin"
        chords_weight = torch.load(chords_ckpt, map_location="cpu")
        self.chords_model.load_state_dict(chords_weight)

    def generate_beats(self, prompt):
        tokenized = self.beats_tokenizer(
            prompt, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        tokenized = {k: v.to(self.beats_model.device) for k, v in tokenized.items()}

        with torch.no_grad():
            out = self.beats_model(**tokenized)

        max_beat = (
            1 + torch.argmax(out["logits"][:, 0, :], -1).detach().cpu().numpy()
        ).tolist()[0]
        intervals = (
            out["values"][:, :, 0]
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
            .round(4)
            .tolist()
        )

        intervals = np.cumsum(intervals)
        predicted_beats_times = []
        for t in intervals:
            if t < 10:
                predicted_beats_times.append(round(t, 2))
            else:
                break
        predicted_beats_times = list(np.array(predicted_beats_times)[:50])

        if len(predicted_beats_times) == 0:
            predicted_beats = [[], []]
        else:
            beat_counts = []
            for i in range(len(predicted_beats_times)):
                beat_counts.append(float(1.0 + np.mod(i, max_beat)))
            predicted_beats = [[predicted_beats_times, beat_counts]]

        return max_beat, predicted_beats_times, predicted_beats

    def generate(self, prompt):
        max_beat, predicted_beats_times, predicted_beats = self.generate_beats(prompt)

        chords_prompt = "Caption: {} \\n Timestamps: {} \\n Max Beat: {}".format(
            prompt,
            " , ".join([str(round(t, 2)) for t in predicted_beats_times]),
            max_beat,
        )

        tokenized = self.chords_tokenizer(
            chords_prompt,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.chords_model.device) for k, v in tokenized.items()}

        generated_chords = self.chords_model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            min_length=8,
            max_length=128,
            num_beams=5,
            early_stopping=True,
            num_return_sequences=1,
        )

        generated_chords = self.chords_tokenizer.decode(
            generated_chords[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).split(" n ")

        predicted_chords, predicted_chords_times = [], []
        for item in generated_chords:
            c, ct = item.split(" at ")
            predicted_chords.append(c)
            predicted_chords_times.append(float(ct))

        return predicted_beats, predicted_chords, predicted_chords_times


class Mustango:
    def __init__(
        self,
        name="declare-lab/mustango",
        device="cuda:0",
        cache_dir=None,
        local_files_only=False,
    ):
        path = snapshot_download(repo_id=name, cache_dir=cache_dir)

        self.music_model = MusicFeaturePredictor(
            path, device, cache_dir=cache_dir, local_files_only=local_files_only
        )

        vae_config = json.load(open(f"{path}/configs/vae_config.json"))
        stft_config = json.load(open(f"{path}/configs/stft_config.json"))
        main_config = json.load(open(f"{path}/configs/main_config.json"))

        self.vae = AutoencoderKL(**vae_config).to(device)
        self.stft = TacotronSTFT(**stft_config).to(device)
        self.model = MusicAudioDiffusion(
            main_config["text_encoder_name"],
            main_config["scheduler_name"],
            unet_model_config_path=f"{path}/configs/music_diffusion_model_config.json",
        ).to(device)

        vae_weights = torch.load(
            f"{path}/vae/pytorch_model_vae.bin", map_location=device
        )
        stft_weights = torch.load(
            f"{path}/stft/pytorch_model_stft.bin", map_location=device
        )
        main_weights = torch.load(
            f"{path}/ldm/pytorch_model_ldm.bin", map_location=device
        )

        self.vae.load_state_dict(vae_weights)
        self.stft.load_state_dict(stft_weights)
        self.model.load_state_dict(main_weights)

        print("Successfully loaded checkpoint from:", name)

        self.vae.eval()
        self.stft.eval()
        self.model.eval()

        self.scheduler = DDPMScheduler.from_pretrained(
            main_config["scheduler_name"], subfolder="scheduler"
        )

    def generate(
        self, 
        prompt, 
        steps=100,
        guidance=3, 
        samples=1, 
        disable_progress=True, 
        return_latent_t_dict=False, 
        leading_latents=None, 
        clip_ratio=0.375,
        tail_ratio=0.125,
    ):
        """Genrate music for a single prompt string."""

        with torch.no_grad():
            beats, chords, chords_times = self.music_model.generate(prompt)
            out = self.model.inference(
                [prompt],
                beats,
                [chords],
                [chords_times],
                self.scheduler,
                steps,
                guidance,
                samples,
                disable_progress,
                leading_latents=leading_latents,
                clip_ratio=clip_ratio,
                tail_ratio=tail_ratio,
                return_latent_t_dict=return_latent_t_dict
            )
            if return_latent_t_dict:
                latents, latent_t_dict = out
            else:
                latents = out
            
            mel = self.vae.decode_first_stage(latents)
            wave = self.vae.decode_to_waveform(mel)

        return wave[0], latent_t_dict if return_latent_t_dict else wave[0]
    
    def generate_longer(self, prompt: str, n_secs: int, t_err=0.1, return_slices: bool = True):
        # step.1 compute length
        DEFAULT_LENGTH = 10.242   # do NOT change, original generated length.
        SLICE_GEN_LENGTH = 5.121  # may not change (requires corresponded modification to `clip_ratio`)
        DROP_TAIL = 1 / 8  # may not change (requires corresponded modification to `tail_ratio`)
        
        n_full_length_runs = int((n_secs - DEFAULT_LENGTH) // SLICE_GEN_LENGTH) + 1
        total_diff = n_secs - (n_full_length_runs - 1) * SLICE_GEN_LENGTH - DEFAULT_LENGTH
        
        _clip_ratio = None
        if total_diff > t_err:
            # when time gap > acceptable error
            _clip_ratio = (DEFAULT_LENGTH - total_diff) / DEFAULT_LENGTH - DROP_TAIL

        # step.2 coherent gen
        music, leading_latents = self.generate(prompt=prompt, leading_latents=None, return_latent_t_dict=True)
        musics = [music[:int(len(music) / DEFAULT_LENGTH * SLICE_GEN_LENGTH)]]
        for i in range(n_full_length_runs - 1):            
            if i < n_full_length_runs - 2:
                music, leading_latents = self.generate(
                    prompt=prompt, leading_latents=leading_latents, return_latent_t_dict=True)
                musics.append(music[:int(len(music) / DEFAULT_LENGTH * SLICE_GEN_LENGTH)])
            else:
                if _clip_ratio is None:
                    music, leading_latents = self.generate(
                        prompt=prompt, leading_latents=leading_latents, return_latent_t_dict=True)
                    musics.append(music)
                else:
                    music, leading_latents = self.generate(
                        prompt=prompt, leading_latents=leading_latents, clip_ratio=_clip_ratio, return_latent_t_dict=True)
                    musics.append(music[:int(len(music) * total_diff / DEFAULT_LENGTH)])
        
        if _clip_ratio is not None:
            music, _ = self.generate(prompt=prompt, leading_latents=leading_latents, return_latent_t_dict=True)
            musics.append(music)
            
        return np.concatenate(musics), musics if return_slices else np.concatenate(musics)