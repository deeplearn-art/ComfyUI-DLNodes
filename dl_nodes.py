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
    random UMT5/T5 token conditionings.
    """

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION     = "encode"
    CATEGORY     = "conditioning"

    # -------- sockets ------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "umt5": ("CLIP",),          # loader’s output type
                "n_tokens": ("INT", {
                    "default": 3, "min": 0,
                    "max": 511, "step": 1,
                    "display": "number",
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, *_, **__):
        return float("nan")

    # -------- helpers ------------------------------------------------
    @staticmethod
    def _get_vocab_size(tok):
        """Return int vocab size even if property is missing."""
        if hasattr(tok, "vocab_size"):
            attr = getattr(tok, "vocab_size")
            return attr() if callable(attr) else attr
        if hasattr(tok, "get_vocab_size"):
            return tok.get_vocab_size(with_added_tokens=True)
        if hasattr(tok, "get_vocab"):
            return len(tok.get_vocab())
        return 250_112                        # sensible UMT5 default

    @staticmethod
    def _safe_attr(tok, name, default):
        return getattr(tok, name, default)

    # -------- main ---------------------------------------------------
    def encode(self, umt5, n_tokens):
      tok         = umt5.tokenizer
      vocab_size  = self._get_vocab_size(tok)
      pad_id      = getattr(tok, "pad_token_id", 0)
      eos_id      = getattr(tok, "eos_token_id", 1)
      max_len     = getattr(tok, "model_max_length", 512)

      n_tokens = min(n_tokens, max_len - 1)

      ids  = [random.randint(2, vocab_size - 1) for _ in range(n_tokens)]
      ids += [eos_id]
      ids += [pad_id] * (max_len - len(ids))

      token_list = [(tid, 1.0) for tid in ids]

      clip_key = getattr(
          getattr(umt5, "cond_stage_model", umt5),  # inner model if it exists
          "clip_name",
          "l",                                     # safe fallback
      )
      tokens     = {clip_key: [token_list]}          # <- HERE

      conditioning = umt5.encode_from_tokens_scheduled(tokens)
      return (conditioning,)

    
NODE_CLASS_MAPPINGS = {
    "String2List": String2ListNode,
    "CLIPRandom": CLIPRandom,
    "UMT5Random": UMT5Random
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "String2List": "String2List",
    "CLIPRandom": "Random CLIP Tokens",
    "UMT5Random": "Random UMT5 Tokens"
}



