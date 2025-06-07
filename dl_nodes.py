import random, torch

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
            },
        }

    @classmethod
    def IS_CHANGED():
        return float("nan")

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, n_tokens):
      
        token_list = [(49406, 1.0)]
        for n in range(n_tokens):
            token_list.append((random.randint(0, 49405),1.0))
        for n in range(n_tokens,76):
             token_list.append((49407, 1.0))
        #breakpoint()
        tokens = {'l': [token_list]}
        #print(f" Using tokens {t1},{t2},{t3} " )
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )
        

class UMT5Random:
    """
    Random UMT5/T5 conditioning generator.

    Widgets
    ▸ n_tokens …… number of random IDs (2…vocab_size-1)
    ▸ padding   …… number of explicit <pad> (=0⃗) tokens to append
    ▸ random_weights …… if true, sample weights in the range [-1, +1]
    ▸ use_anchor …… include <eos> after the random IDs
    """

    # two outputs: conditioning + a human-readable string
    RETURN_TYPES  = ("CONDITIONING", "STRING")
    RETURN_NAMES  = ("conditioning", "tokens_view")
    FUNCTION      = "encode"
    CATEGORY      = "conditioning"

    # -------- sockets & widgets -------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "umt5": ("CLIP",),
                "n_tokens": ("INT", {
                    "default": 3, "min": 0,
                    "max": 511, "step": 1,
                    "display": "number",
                }),
                "padding": ("INT", {                   # NEW
                    "default": 0, "min": 0,
                    "max": 511, "step": 1,
                    "display": "number",
                    "tooltip": "How many <pad> (zero) tokens to append",
                }),
                "random_weights": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, sample token weights in [-1, 1]",
                }),
                "use_anchor": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If false, drops the <eos> anchor token",
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, *_, **__):
        return float("nan")

    # -------- helpers ------------------------------------------------
    @staticmethod
    def _get_vocab_size(tok):
        if hasattr(tok, "vocab_size"):
            attr = getattr(tok, "vocab_size")
            return attr() if callable(attr) else attr
        if hasattr(tok, "get_vocab_size"):
            return tok.get_vocab_size(with_added_tokens=True)
        if hasattr(tok, "get_vocab"):
            return len(tok.get_vocab())
        return 250_112

    # -------- main ---------------------------------------------------
    def encode(
        self,
        umt5,
        n_tokens: int,
        padding: int,
        random_weights: bool = False,
        use_anchor: bool = True,
    ):
        tok         = umt5.tokenizer
        vocab_size  = self._get_vocab_size(tok)
        pad_id      = getattr(tok, "pad_token_id", 0)
        eos_id      = getattr(tok, "eos_token_id", 1)
        model_max   = getattr(tok, "model_max_length", 512)

        extra = 1 if use_anchor else 0
        max_len = n_tokens + extra + padding

        # stop runaway values
        if max_len > model_max:
            overflow = max_len - model_max
            # trim padding first, then random IDs
            trim_pad = min(padding, overflow)
            padding -= trim_pad
            overflow -= trim_pad
            n_tokens = max(0, n_tokens - overflow)
            max_len  = model_max

        # ---- assemble ID list ---------------------------------------
        ids = [random.randint(2, vocab_size - 1) for _ in range(n_tokens)]
        if use_anchor:
            ids.append(eos_id)
        ids += [pad_id] * padding

        # ---- assign weights -----------------------------------------
        if random_weights:
            token_list = [(tid, random.uniform(-1.0, 1.0)) for tid in ids]
        else:
            token_list = [(tid, 1.0) for tid in ids]

        # ---- conditioning dict key ----------------------------------
        clip_key = getattr(getattr(umt5, "cond_stage_model", umt5),
                           "clip_name", "l")
        tokens = {clip_key: [token_list]}

        conditioning = umt5.encode_from_tokens_scheduled(tokens)

        # ---- pretty print for UI -----------------------------------
        id_strings = tok.convert_ids_to_tokens(ids, skip_special_tokens=False)
        if random_weights:
            pretty = " ".join(
                f"{t}({w:+.2f})"
                for (t, w) in zip(id_strings, (tw[1] for tw in token_list))
            )
        else:
            pretty = " ".join(id_strings)

        return (conditioning, pretty)


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



