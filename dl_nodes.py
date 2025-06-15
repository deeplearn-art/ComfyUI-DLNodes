import torch 
import random
class String2ListNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING",  {"forceInput": True}),
                "separator": ("STRING", {"default": "/"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("TEXT",)
    #OUTPUT_IS_LIST = (True, )
    FUNCTION = "perform_split_string"
    CATEGORY = "DLNodes"

    def perform_split_string(self, text, separator):
        str_list = text.split(separator)
        ret_arr = []
        for s in str_list:
            ret_arr += [s]
            
        return (ret_arr, )



class CLIPRandom:
    """
    random clip tokens
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "n_tokens": ("INT", {
                    "default": 3, 
                    "min": 0, #Minimum value
                    "max": 75, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "random_weights": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, n_tokens, random_weights=False):
        # Build token id list: SOS (49406) + random tokens + padding tokens (49407)
        token_ids = [49406] + [random.randint(0, 49405) for _ in range(n_tokens)] + [49407] * (76 - n_tokens)

        # Attach weights
        if random_weights:
            token_list = [(tid, random.uniform(-1.0, 1.0)) for tid in token_ids]
        else:
            token_list = [(tid, 1.0) for tid in token_ids]

        # primary tokens dictionary for typical SD 1.x CLIP
        tokens = {"l": [token_list]}

        # Try encoding; if model expects both "l" and "g" (e.g., SDXL) fall back
        try:
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        except KeyError:
            # duplicate list for "g" branch required by SDXL CLIP implementation
            tokens["g"] = [token_list]
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        return ([[cond, {"pooled_output": pooled}]], )
        
class UMT5Random:
    RETURN_TYPES  = ("CONDITIONING", "STRING")
    RETURN_NAMES  = ("conditioning", "tokens_view")
    FUNCTION      = "encode"
    CATEGORY      = "conditioning"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "umt5": ("CLIP",),
                "n_tokens": ("INT", {"default": 3, "min": 0, "max": 511}),
                "padding":  ("INT", {"default": 0, "min": 0, "max": 511}),
                "random_weights": ("BOOLEAN", {"default": False}),
                "use_anchor":     ("BOOLEAN", {"default": True}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, *_, **__):
        return float("nan")

    # ---------------- helpers -------------------------------------
    @staticmethod
    def _get_vocab_size(tok):
        if hasattr(tok, "vocab_size"):
            v = tok.vocab_size
            return v() if callable(v) else v
        if hasattr(tok, "get_vocab_size"):
            return tok.get_vocab_size(with_added_tokens=True)
        if hasattr(tok, "get_vocab"):
            return len(tok.get_vocab())
        return 250_112                                # safe default

    @staticmethod
    def _ids_to_tokens(tok, ids):
        """
        Robustly map list[int] → list[str] for any tokenizer wrapper.
        """
        if hasattr(tok, "convert_ids_to_tokens"):
            return tok.convert_ids_to_tokens(ids, skip_special_tokens=False)

        out = []
        for i in ids:
            if hasattr(tok, "id_to_token"):               # tokenizers lib
                out.append(tok.id_to_token(i))
            else:
                try:                                      # fallback decode
                    out.append(tok.decode([i]).strip() or f"<{i}>")
                except Exception:
                    out.append(f"<{i}>")
        return out

    # ---------------- main ----------------------------------------
    def encode(
        self,
        umt5,
        n_tokens: int,
        padding: int,
        random_weights: bool = False,
        use_anchor: bool = True,
    ):
        tok        = umt5.tokenizer
        vocab_size = self._get_vocab_size(tok)
        pad_id     = getattr(tok, "pad_token_id", 0)
        eos_id     = getattr(tok, "eos_token_id", 1)
        model_max  = getattr(tok, "model_max_length", 512)

        extra    = 1 if use_anchor else 0
        max_len  = n_tokens + padding + extra

        if max_len > model_max:                # trim if user overshoots
            overflow  = max_len - model_max
            trim_pad  = min(padding, overflow)
            padding  -= trim_pad
            overflow -= trim_pad
            n_tokens  = max(0, n_tokens - overflow)
            max_len   = model_max

        # ----------- build IDs ------------------------------------
        ids = [random.randint(2, vocab_size - 1) for _ in range(n_tokens)]
        if use_anchor:
            ids.append(eos_id)
        ids += [pad_id] * padding

        # ----------- weights --------------------------------------
        if random_weights:
            token_list = [(tid, random.uniform(-1.0, 1.0)) for tid in ids]
        else:
            token_list = [(tid, 1.0) for tid in ids]

        # ----------- conditioning dict key ------------------------
        clip_key = getattr(getattr(umt5, "cond_stage_model", umt5),
                           "clip_name", "l")
        tokens_dict = {clip_key: [token_list]}
        conditioning = umt5.encode_from_tokens_scheduled(tokens_dict)

        # ----------- pretty print ---------------------------------
        token_strings = self._ids_to_tokens(tok, ids)
        if random_weights:
            view = " ".join(
                f"{t}({w:+.2f})" for t, (_, w) in zip(token_strings, token_list)
            )
        else:
            view = " ".join(token_strings)

        return (conditioning, view)


NODE_CLASS_MAPPINGS        = {"UMT5Random": UMT5Random}
NODE_DISPLAY_NAME_MAPPINGS = {"UMT5Random": "Random UMT5 Tokens"}


NODE_CLASS_MAPPINGS = {
    "String2List": String2ListNode,
    "CLIPRandom": CLIPRandom,
    "UMT5Random": UMT5Random
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "String2List": "String2List",
    "CLIPRandom": "Random CLIP Tokens",
    "UMT5Random": "Random UMT5 Tokens"
}



